// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	"golang.org/x/mod/modfile"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/lsp/source"
	workfile "golang.org/x/tools/internal/mod/modfile"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/xcontext"
	errors "golang.org/x/xerrors"
)

type workspaceSource int

const (
	legacyWorkspace = iota
	goplsModWorkspace
	goWorkWorkspace
	fileSystemWorkspace
)

func (s workspaceSource) String() string {
	switch s {
	case legacyWorkspace:
		return "legacy"
	case goplsModWorkspace:
		return "gopls.mod"
	case goWorkWorkspace:
		return "go.work"
	case fileSystemWorkspace:
		return "file system"
	default:
		return "!(unknown module source)"
	}
}

// workspace tracks go.mod files in the workspace, along with the
// gopls.mod file, to provide support for multi-module workspaces.
//
// Specifically, it provides:
//  - the set of modules contained within in the workspace root considered to
//    be 'active'
//  - the workspace modfile, to be used for the go command `-modfile` flag
//  - the set of workspace directories
//
// This type is immutable (or rather, idempotent), so that it may be shared
// across multiple snapshots.
type workspace struct {
	root         span.URI
	excludePath  func(string) bool
	moduleSource workspaceSource

	// activeModFiles holds the active go.mod files.
	activeModFiles map[span.URI]struct{}

	// knownModFiles holds the set of all go.mod files in the workspace.
	// In all modes except for legacy, this is equivalent to modFiles.
	knownModFiles map[span.URI]struct{}

	// go111moduleOff indicates whether GO111MODULE=off has been configured in
	// the environment.
	go111moduleOff bool

	// The workspace module is lazily re-built once after being invalidated.
	// buildMu+built guards this reconstruction.
	//
	// file and wsDirs may be non-nil even if built == false, if they were copied
	// from the previous workspace module version. In this case, they will be
	// preserved if building fails.
	buildMu  sync.Mutex
	built    bool
	buildErr error
	mod      *modfile.File
	sum      []byte
	wsDirs   map[span.URI]struct{}
}

func newWorkspace(ctx context.Context, root span.URI, fs source.FileSource, excludePath func(string) bool, go111moduleOff bool, experimental bool) (*workspace, error) {
	// The user may have a gopls.mod or go.work file that defines their
	// workspace.
	ws, err := parseExplicitWorkspaceFile(ctx, root, fs, excludePath)
	if err == nil {
		return ws, nil
	}
	// Otherwise, in all other modes, search for all of the go.mod files in the
	// workspace.
	knownModFiles, err := findModules(root, excludePath, 0)
	if err != nil {
		return nil, err
	}
	// When GO111MODULE=off, there are no active go.mod files.
	if go111moduleOff {
		return &workspace{
			root:           root,
			excludePath:    excludePath,
			moduleSource:   legacyWorkspace,
			knownModFiles:  knownModFiles,
			go111moduleOff: true,
		}, nil
	}
	// In legacy mode, not all known go.mod files will be considered active.
	if !experimental {
		activeModFiles, err := getLegacyModules(ctx, root, fs)
		if err != nil {
			return nil, err
		}
		return &workspace{
			root:           root,
			excludePath:    excludePath,
			activeModFiles: activeModFiles,
			knownModFiles:  knownModFiles,
			moduleSource:   legacyWorkspace,
		}, nil
	}
	return &workspace{
		root:           root,
		excludePath:    excludePath,
		activeModFiles: knownModFiles,
		knownModFiles:  knownModFiles,
		moduleSource:   fileSystemWorkspace,
	}, nil
}

func parseExplicitWorkspaceFile(ctx context.Context, root span.URI, fs source.FileSource, excludePath func(string) bool) (*workspace, error) {
	for _, src := range []workspaceSource{goWorkWorkspace, goplsModWorkspace} {
		fh, err := fs.GetFile(ctx, uriForSource(root, src))
		if err != nil {
			return nil, err
		}
		contents, err := fh.Read()
		if err != nil {
			continue
		}
		var file *modfile.File
		var activeModFiles map[span.URI]struct{}
		switch src {
		case goWorkWorkspace:
			file, activeModFiles, err = parseGoWork(ctx, root, fh.URI(), contents, fs)
		case goplsModWorkspace:
			file, activeModFiles, err = parseGoplsMod(root, fh.URI(), contents)
		}
		if err != nil {
			return nil, err
		}
		return &workspace{
			root:           root,
			excludePath:    excludePath,
			activeModFiles: activeModFiles,
			knownModFiles:  activeModFiles,
			mod:            file,
			moduleSource:   src,
		}, nil
	}
	return nil, noHardcodedWorkspace
}

var noHardcodedWorkspace = errors.New("no hardcoded workspace")

func (w *workspace) getKnownModFiles() map[span.URI]struct{} {
	return w.knownModFiles
}

func (w *workspace) getActiveModFiles() map[span.URI]struct{} {
	return w.activeModFiles
}

// modFile gets the workspace modfile associated with this workspace,
// computing it if it doesn't exist.
//
// A fileSource must be passed in to solve a chicken-egg problem: it is not
// correct to pass in the snapshot file source to newWorkspace when
// invalidating, because at the time these are called the snapshot is locked.
// So we must pass it in later on when actually using the modFile.
func (w *workspace) modFile(ctx context.Context, fs source.FileSource) (*modfile.File, error) {
	w.build(ctx, fs)
	return w.mod, w.buildErr
}

func (w *workspace) sumFile(ctx context.Context, fs source.FileSource) ([]byte, error) {
	w.build(ctx, fs)
	return w.sum, w.buildErr
}

func (w *workspace) build(ctx context.Context, fs source.FileSource) {
	w.buildMu.Lock()
	defer w.buildMu.Unlock()

	if w.built {
		return
	}
	// Building should never be cancelled. Since the workspace module is shared
	// across multiple snapshots, doing so would put us in a bad state, and it
	// would not be obvious to the user how to recover.
	ctx = xcontext.Detach(ctx)

	// If our module source is not gopls.mod, try to build the workspace module
	// from modules. Fall back on the pre-existing mod file if parsing fails.
	if w.moduleSource != goplsModWorkspace {
		file, err := buildWorkspaceModFile(ctx, w.activeModFiles, fs)
		switch {
		case err == nil:
			w.mod = file
		case w.mod != nil:
			// Parsing failed, but we have a previous file version.
			event.Error(ctx, "building workspace mod file", err)
		default:
			// No file to fall back on.
			w.buildErr = err
		}
	}
	if w.mod != nil {
		w.wsDirs = map[span.URI]struct{}{
			w.root: {},
		}
		for _, r := range w.mod.Replace {
			// We may be replacing a module with a different version, not a path
			// on disk.
			if r.New.Version != "" {
				continue
			}
			w.wsDirs[span.URIFromPath(r.New.Path)] = struct{}{}
		}
	}
	// Ensure that there is always at least the root dir.
	if len(w.wsDirs) == 0 {
		w.wsDirs = map[span.URI]struct{}{
			w.root: {},
		}
	}
	sum, err := buildWorkspaceSumFile(ctx, w.activeModFiles, fs)
	if err == nil {
		w.sum = sum
	} else {
		event.Error(ctx, "building workspace sum file", err)
	}
	w.built = true
}

// dirs returns the workspace directories for the loaded modules.
func (w *workspace) dirs(ctx context.Context, fs source.FileSource) []span.URI {
	w.build(ctx, fs)
	var dirs []span.URI
	for d := range w.wsDirs {
		dirs = append(dirs, d)
	}
	sort.Slice(dirs, func(i, j int) bool {
		return source.CompareURI(dirs[i], dirs[j]) < 0
	})
	return dirs
}

// invalidate returns a (possibly) new workspace after invalidating the changed
// files. If w is still valid in the presence of changedURIs, it returns itself
// unmodified.
//
// The returned changed and reload flags control the level of invalidation.
// Some workspace changes may affect workspace contents without requiring a
// reload of metadata (for example, unsaved changes to a go.mod or go.sum
// file).
func (w *workspace) invalidate(ctx context.Context, changes map[span.URI]*fileChange, fs source.FileSource) (_ *workspace, changed, reload bool) {
	// Prevent races to w.modFile or w.wsDirs below, if wmhas not yet been built.
	w.buildMu.Lock()
	defer w.buildMu.Unlock()

	// Clone the workspace. This may be discarded if nothing changed.
	result := &workspace{
		root:           w.root,
		moduleSource:   w.moduleSource,
		knownModFiles:  make(map[span.URI]struct{}),
		activeModFiles: make(map[span.URI]struct{}),
		go111moduleOff: w.go111moduleOff,
		mod:            w.mod,
		sum:            w.sum,
		wsDirs:         w.wsDirs,
	}
	for k, v := range w.knownModFiles {
		result.knownModFiles[k] = v
	}
	for k, v := range w.activeModFiles {
		result.activeModFiles[k] = v
	}

	// First handle changes to the go.work or gopls.mod file. This must be
	// considered before any changes to go.mod or go.sum files, as these files
	// determine which modules we care about. If go.work/gopls.mod has changed
	// we need to either re-read it if it exists or walk the filesystem if it
	// has been deleted. go.work should override the gopls.mod if both exist.
	if changedInner, reloadInner, found := updateExplicitWorkspaceFile(ctx, w, result, changes, fs); found {
		changed = changedInner
		reload = reloadInner
	}
	// Next, handle go.mod changes that could affect our workspace. If we're
	// reading our tracked modules from the gopls.mod, there's nothing to do
	// here.
	if result.moduleSource != goplsModWorkspace && result.moduleSource != goWorkWorkspace {
		for uri, change := range changes {
			// Otherwise, we only care about go.mod files in the workspace directory.
			if change.isUnchanged || !isGoMod(uri) || !source.InDir(result.root.Filename(), uri.Filename()) {
				continue
			}
			changed = true
			active := result.moduleSource != legacyWorkspace || source.CompareURI(modURI(w.root), uri) == 0
			reload = reload || (active && change.fileHandle.Saved())
			if change.exists {
				result.knownModFiles[uri] = struct{}{}
				if active {
					result.activeModFiles[uri] = struct{}{}
				}
			} else {
				delete(result.knownModFiles, uri)
				delete(result.activeModFiles, uri)
			}
		}
	}

	// Finally, process go.sum changes for any modules that are now active.
	for uri, change := range changes {
		if !isGoSum(uri) {
			continue
		}
		// TODO(rFindley) factor out this URI mangling.
		dir := filepath.Dir(uri.Filename())
		modURI := span.URIFromPath(filepath.Join(dir, "go.mod"))
		if _, active := result.activeModFiles[modURI]; !active {
			continue
		}
		// Only changes to active go.sum files actually cause the workspace to
		// change.
		changed = true
		reload = reload || change.fileHandle.Saved()
	}

	if !changed {
		return w, false, false
	}

	return result, changed, reload
}

// updateExplicitWorkspaceFile checks if any of the changes happened to a go.work or
// gopls.mod file, and if so, updates the result accordingly.
func updateExplicitWorkspaceFile(ctx context.Context, w, result *workspace, changes map[span.URI]*fileChange, fs source.FileSource) (changed, reload, found bool) {
	// If go.work/gopls.mod has changed we need to either re-read it if it
	// exists or walk the filesystem if it has been deleted.
	// go.work should override the gopls.mod if both exist.
	for _, src := range []workspaceSource{goplsModWorkspace, goWorkWorkspace} {
		uri := uriForSource(w.root, src)
		// File opens/closes are just no-ops.
		change, ok := changes[uri]
		if !ok || change.isUnchanged {
			continue
		}
		found = true
		if change.exists {
			// Only invalidate if the file if it actually parses.
			// Otherwise, stick with the current file.
			var parsedFile *modfile.File
			var parsedModules map[span.URI]struct{}
			var err error
			switch src {
			case goWorkWorkspace:
				parsedFile, parsedModules, err = parseGoWork(ctx, w.root, uri, change.content, fs)
			case goplsModWorkspace:
				parsedFile, parsedModules, err = parseGoplsMod(w.root, uri, change.content)
			}
			if err != nil {
				// An unparseable file should not invalidate the workspace:
				// nothing good could come from changing the workspace in
				// this case.
				event.Error(ctx, fmt.Sprintf("parsing %s", filepath.Base(uri.Filename())), err)
			}
			changed = true
			reload = change.fileHandle.Saved()
			result.mod = parsedFile
			result.moduleSource = src
			result.knownModFiles = parsedModules
			result.activeModFiles = make(map[span.URI]struct{})
			for k, v := range parsedModules {
				result.activeModFiles[k] = v
			}
		} else {
			// go.work/gopls.mod is deleted. search for modules again.
			changed = true
			reload = true
			result.moduleSource = fileSystemWorkspace
			// The parsed file is no longer valid.
			result.mod = nil
			knownModFiles, err := findModules(w.root, w.excludePath, 0)
			if err != nil {
				result.knownModFiles = nil
				result.activeModFiles = nil
				event.Error(ctx, "finding file system modules", err)
			} else {
				result.knownModFiles = knownModFiles
				result.activeModFiles = make(map[span.URI]struct{})
				for k, v := range result.knownModFiles {
					result.activeModFiles[k] = v
				}
			}
		}
	}
	return changed, reload, found
}

// goplsModURI returns the URI for the gopls.mod file contained in root.
func uriForSource(root span.URI, src workspaceSource) span.URI {
	var basename string
	switch src {
	case goplsModWorkspace:
		basename = "gopls.mod"
	case goWorkWorkspace:
		basename = "go.work"
	default:
		return ""
	}
	return span.URIFromPath(filepath.Join(root.Filename(), basename))
}

// modURI returns the URI for the go.mod file contained in root.
func modURI(root span.URI) span.URI {
	return span.URIFromPath(filepath.Join(root.Filename(), "go.mod"))
}

// isGoMod reports if uri is a go.mod file.
func isGoMod(uri span.URI) bool {
	return filepath.Base(uri.Filename()) == "go.mod"
}

func isGoSum(uri span.URI) bool {
	return filepath.Base(uri.Filename()) == "go.sum"
}

// fileExists reports if the file uri exists within source.
func fileExists(ctx context.Context, uri span.URI, source source.FileSource) (bool, error) {
	fh, err := source.GetFile(ctx, uri)
	if err != nil {
		return false, err
	}
	return fileHandleExists(fh)
}

// fileHandleExists reports if the file underlying fh actually exits.
func fileHandleExists(fh source.FileHandle) (bool, error) {
	_, err := fh.Read()
	if err == nil {
		return true, nil
	}
	if os.IsNotExist(err) {
		return false, nil
	}
	return false, err
}

// TODO(rFindley): replace this (and similar) with a uripath package analogous
// to filepath.
func dirURI(uri span.URI) span.URI {
	return span.URIFromPath(filepath.Dir(uri.Filename()))
}

// getLegacyModules returns a module set containing at most the root module.
func getLegacyModules(ctx context.Context, root span.URI, fs source.FileSource) (map[span.URI]struct{}, error) {
	uri := span.URIFromPath(filepath.Join(root.Filename(), "go.mod"))
	modules := make(map[span.URI]struct{})
	exists, err := fileExists(ctx, uri, fs)
	if err != nil {
		return nil, err
	}
	if exists {
		modules[uri] = struct{}{}
	}
	return modules, nil
}

func parseGoWork(ctx context.Context, root, uri span.URI, contents []byte, fs source.FileSource) (*modfile.File, map[span.URI]struct{}, error) {
	workFile, err := workfile.ParseWork(uri.Filename(), contents, nil)
	if err != nil {
		return nil, nil, errors.Errorf("parsing go.work: %w", err)
	}
	modFiles := make(map[span.URI]struct{})
	for _, dir := range workFile.Directory {
		// The resulting modfile must use absolute paths, so that it can be
		// written to a temp directory.
		dir.DiskPath = absolutePath(root, dir.DiskPath)
		modURI := span.URIFromPath(filepath.Join(dir.DiskPath, "go.mod"))
		modFiles[modURI] = struct{}{}
	}
	modFile, err := buildWorkspaceModFile(ctx, modFiles, fs)
	if err != nil {
		return nil, nil, err
	}
	if workFile.Go.Version != "" {
		if err := modFile.AddGoStmt(workFile.Go.Version); err != nil {
			return nil, nil, err
		}
	}

	return modFile, modFiles, nil
}

func parseGoplsMod(root, uri span.URI, contents []byte) (*modfile.File, map[span.URI]struct{}, error) {
	modFile, err := modfile.Parse(uri.Filename(), contents, nil)
	if err != nil {
		return nil, nil, errors.Errorf("parsing gopls.mod: %w", err)
	}
	modFiles := make(map[span.URI]struct{})
	for _, replace := range modFile.Replace {
		if replace.New.Version != "" {
			return nil, nil, errors.Errorf("gopls.mod: replaced module %q@%q must not have version", replace.New.Path, replace.New.Version)
		}
		// The resulting modfile must use absolute paths, so that it can be
		// written to a temp directory.
		replace.New.Path = absolutePath(root, replace.New.Path)
		modURI := span.URIFromPath(filepath.Join(replace.New.Path, "go.mod"))
		modFiles[modURI] = struct{}{}
	}
	return modFile, modFiles, nil
}

func absolutePath(root span.URI, path string) string {
	dirFP := filepath.FromSlash(path)
	if !filepath.IsAbs(dirFP) {
		dirFP = filepath.Join(root.Filename(), dirFP)
	}
	return dirFP
}

// errExhausted is returned by findModules if the file scan limit is reached.
var errExhausted = errors.New("exhausted")

// Limit go.mod search to 1 million files. As a point of reference,
// Kubernetes has 22K files (as of 2020-11-24).
const fileLimit = 1000000

// findModules recursively walks the root directory looking for go.mod files,
// returning the set of modules it discovers. If modLimit is non-zero,
// searching stops once modLimit modules have been found.
//
// TODO(rfindley): consider overlays.
func findModules(root span.URI, excludePath func(string) bool, modLimit int) (map[span.URI]struct{}, error) {
	// Walk the view's folder to find all modules in the view.
	modFiles := make(map[span.URI]struct{})
	searched := 0
	errDone := errors.New("done")
	err := filepath.Walk(root.Filename(), func(path string, info os.FileInfo, err error) error {
		if err != nil {
			// Probably a permission error. Keep looking.
			return filepath.SkipDir
		}
		// For any path that is not the workspace folder, check if the path
		// would be ignored by the go command. Vendor directories also do not
		// contain workspace modules.
		if info.IsDir() && path != root.Filename() {
			suffix := strings.TrimPrefix(path, root.Filename())
			switch {
			case checkIgnored(suffix),
				strings.Contains(filepath.ToSlash(suffix), "/vendor/"),
				excludePath(suffix):
				return filepath.SkipDir
			}
		}
		// We're only interested in go.mod files.
		uri := span.URIFromPath(path)
		if isGoMod(uri) {
			modFiles[uri] = struct{}{}
		}
		if modLimit > 0 && len(modFiles) >= modLimit {
			return errDone
		}
		searched++
		if fileLimit > 0 && searched >= fileLimit {
			return errExhausted
		}
		return nil
	})
	if err == errDone {
		return modFiles, nil
	}
	return modFiles, err
}

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"context"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	"golang.org/x/mod/modfile"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/xcontext"
	errors "golang.org/x/xerrors"
)

type workspaceSource int

const (
	legacyWorkspace = iota
	goplsModWorkspace
	fileSystemWorkspace
)

func (s workspaceSource) String() string {
	switch s {
	case legacyWorkspace:
		return "legacy"
	case goplsModWorkspace:
		return "gopls.mod"
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

func newWorkspace(ctx context.Context, root span.URI, fs source.FileSource, go111moduleOff bool, experimental bool) (*workspace, error) {
	// In experimental mode, the user may have a gopls.mod file that defines
	// their workspace.
	if experimental {
		goplsModFH, err := fs.GetFile(ctx, goplsModURI(root))
		if err != nil {
			return nil, err
		}
		contents, err := goplsModFH.Read()
		if err == nil {
			file, activeModFiles, err := parseGoplsMod(root, goplsModFH.URI(), contents)
			if err != nil {
				return nil, err
			}
			return &workspace{
				root:           root,
				activeModFiles: activeModFiles,
				knownModFiles:  activeModFiles,
				mod:            file,
				moduleSource:   goplsModWorkspace,
			}, nil
		}
	}
	// Otherwise, in all other modes, search for all of the go.mod files in the
	// workspace.
	knownModFiles, err := findModules(ctx, root, 0)
	if err != nil {
		return nil, err
	}
	// When GO111MODULE=off, there are no active go.mod files.
	if go111moduleOff {
		return &workspace{
			root:           root,
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
			activeModFiles: activeModFiles,
			knownModFiles:  knownModFiles,
			moduleSource:   legacyWorkspace,
		}, nil
	}
	return &workspace{
		root:           root,
		activeModFiles: knownModFiles,
		knownModFiles:  knownModFiles,
		moduleSource:   fileSystemWorkspace,
	}, nil
}

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

// invalidate returns a (possibly) new workspaceModule after invalidating
// changedURIs. If wm is still valid in the presence of changedURIs, it returns
// itself unmodified.
func (w *workspace) invalidate(ctx context.Context, changes map[span.URI]*fileChange) (*workspace, bool) {
	// Prevent races to wm.modFile or wm.wsDirs below, if wm has not yet been
	// built.
	w.buildMu.Lock()
	defer w.buildMu.Unlock()

	// Any gopls.mod change is processed first, followed by go.mod changes, as
	// changes to gopls.mod may affect the set of active go.mod files.
	var (
		// New values. We return a new workspace module if and only if
		// knownModFiles is non-nil.
		knownModFiles map[span.URI]struct{}
		moduleSource  = w.moduleSource
		modFile       = w.mod
		sumData       = w.sum
		err           error
	)
	if w.moduleSource == goplsModWorkspace {
		// If we are currently reading the modfile from gopls.mod, we default to
		// preserving it even if module metadata changes (which may be the case if
		// a go.sum file changes).
		modFile = w.mod
	}
	// First handle changes to the gopls.mod file.
	if w.moduleSource != legacyWorkspace {
		// If gopls.mod has changed we need to either re-read it if it exists or
		// walk the filesystem if it doesn't exist.
		gmURI := goplsModURI(w.root)
		if change, ok := changes[gmURI]; ok {
			if change.exists {
				// Only invalidate if the gopls.mod actually parses. Otherwise, stick with the current gopls.mod
				parsedFile, parsedModules, err := parseGoplsMod(w.root, gmURI, change.content)
				if err == nil {
					modFile = parsedFile
					moduleSource = goplsModWorkspace
					knownModFiles = parsedModules
				} else {
					// Note that modFile is not invalidated here.
					event.Error(ctx, "parsing gopls.mod", err)
				}
			} else {
				// gopls.mod is deleted. search for modules again.
				moduleSource = fileSystemWorkspace
				knownModFiles, err = findModules(ctx, w.root, 0)
				// the modFile is no longer valid.
				if err != nil {
					event.Error(ctx, "finding file system modules", err)
				}
				modFile = nil
			}
		}
	}

	// Next, handle go.mod changes that could affect our set of tracked modules.
	// If we're reading our tracked modules from the gopls.mod, there's nothing
	// to do here.
	if w.moduleSource != goplsModWorkspace {
		for uri, change := range changes {
			// If a go.mod file has changed, we may need to update the set of active
			// modules.
			if !isGoMod(uri) {
				continue
			}
			if !source.InDir(w.root.Filename(), uri.Filename()) {
				// Otherwise, the module must be contained within the workspace root.
				continue
			}
			if knownModFiles == nil {
				knownModFiles = make(map[span.URI]struct{})
				for k := range w.knownModFiles {
					knownModFiles[k] = struct{}{}
				}
			}
			if change.exists {
				knownModFiles[uri] = struct{}{}
			} else {
				delete(knownModFiles, uri)
			}
		}
	}
	if knownModFiles != nil {
		var activeModFiles map[span.URI]struct{}
		if w.go111moduleOff {
			// If GO111MODULE=off, the set of active go.mod files is unchanged.
			activeModFiles = w.activeModFiles
		} else {
			activeModFiles = make(map[span.URI]struct{})
			for uri := range knownModFiles {
				// Legacy mode only considers a module a workspace root, so don't
				// update the active go.mod files map.
				if w.moduleSource == legacyWorkspace && source.CompareURI(modURI(w.root), uri) != 0 {
					continue
				}
				activeModFiles[uri] = struct{}{}
			}
		}
		// Any change to modules triggers a new version.
		return &workspace{
			root:           w.root,
			moduleSource:   moduleSource,
			activeModFiles: activeModFiles,
			knownModFiles:  knownModFiles,
			mod:            modFile,
			sum:            sumData,
			wsDirs:         w.wsDirs,
		}, true
	}
	// No change. Just return wm, since it is immutable.
	return w, false
}

// goplsModURI returns the URI for the gopls.mod file contained in root.
func goplsModURI(root span.URI) span.URI {
	return span.URIFromPath(filepath.Join(root.Filename(), "gopls.mod"))
}

// modURI returns the URI for the go.mod file contained in root.
func modURI(root span.URI) span.URI {
	return span.URIFromPath(filepath.Join(root.Filename(), "go.mod"))
}

// isGoMod reports if uri is a go.mod file.
func isGoMod(uri span.URI) bool {
	return filepath.Base(uri.Filename()) == "go.mod"
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
		dirFP := filepath.FromSlash(replace.New.Path)
		if !filepath.IsAbs(dirFP) {
			dirFP = filepath.Join(root.Filename(), dirFP)
			// The resulting modfile must use absolute paths, so that it can be
			// written to a temp directory.
			replace.New.Path = dirFP
		}
		modURI := span.URIFromPath(filepath.Join(dirFP, "go.mod"))
		modFiles[modURI] = struct{}{}
	}
	return modFile, modFiles, nil
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
func findModules(ctx context.Context, root span.URI, modLimit int) (map[span.URI]struct{}, error) {
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
				strings.Contains(filepath.ToSlash(suffix), "/vendor/"):
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

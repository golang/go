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

	// modFiles holds the active go.mod files.
	modFiles map[span.URI]struct{}

	// The workspace module is lazily re-built once after being invalidated.
	// buildMu+built guards this reconstruction.
	//
	// file and wsDirs may be non-nil even if built == false, if they were copied
	// from the previous workspace module version. In this case, they will be
	// preserved if building fails.
	buildMu  sync.Mutex
	built    bool
	buildErr error
	file     *modfile.File
	wsDirs   map[span.URI]struct{}
}

func newWorkspace(ctx context.Context, root span.URI, fs source.FileSource, experimental bool) (*workspace, error) {
	if !experimental {
		modFiles, err := getLegacyModules(ctx, root, fs)
		if err != nil {
			return nil, err
		}
		return &workspace{
			root:         root,
			modFiles:     modFiles,
			moduleSource: legacyWorkspace,
		}, nil
	}
	goplsModFH, err := fs.GetFile(ctx, goplsModURI(root))
	if err != nil {
		return nil, err
	}
	contents, err := goplsModFH.Read()
	if err == nil {
		file, modFiles, err := parseGoplsMod(root, goplsModFH.URI(), contents)
		if err != nil {
			return nil, err
		}
		return &workspace{
			root:         root,
			modFiles:     modFiles,
			file:         file,
			moduleSource: goplsModWorkspace,
		}, nil
	}
	modFiles, err := findAllModules(ctx, root)
	if err != nil {
		return nil, err
	}
	return &workspace{
		root:         root,
		modFiles:     modFiles,
		moduleSource: fileSystemWorkspace,
	}, nil
}

func (wm *workspace) activeModFiles() map[span.URI]struct{} {
	return wm.modFiles
}

// modFile gets the workspace modfile associated with this workspace,
// computing it if it doesn't exist.
//
// A fileSource must be passed in to solve a chicken-egg problem: it is not
// correct to pass in the snapshot file source to newWorkspace when
// invalidating, because at the time these are called the snapshot is locked.
// So we must pass it in later on when actually using the modFile.
func (wm *workspace) modFile(ctx context.Context, fs source.FileSource) (*modfile.File, error) {
	wm.build(ctx, fs)
	return wm.file, wm.buildErr
}

func (wm *workspace) build(ctx context.Context, fs source.FileSource) {
	wm.buildMu.Lock()
	defer wm.buildMu.Unlock()

	if wm.built {
		return
	}
	// Building should never be cancelled. Since the workspace module is shared
	// across multiple snapshots, doing so would put us in a bad state, and it
	// would not be obvious to the user how to recover.
	ctx = xcontext.Detach(ctx)

	// If our module source is not gopls.mod, try to build the workspace module
	// from modules. Fall back on the pre-existing mod file if parsing fails.
	if wm.moduleSource != goplsModWorkspace {
		file, err := buildWorkspaceModFile(ctx, wm.modFiles, fs)
		switch {
		case err == nil:
			wm.file = file
		case wm.file != nil:
			// Parsing failed, but we have a previous file version.
			event.Error(ctx, "building workspace mod file", err)
		default:
			// No file to fall back on.
			wm.buildErr = err
		}
	}
	if wm.file != nil {
		wm.wsDirs = map[span.URI]struct{}{
			wm.root: {},
		}
		for _, r := range wm.file.Replace {
			// We may be replacing a module with a different version, not a path
			// on disk.
			if r.New.Version != "" {
				continue
			}
			wm.wsDirs[span.URIFromPath(r.New.Path)] = struct{}{}
		}
	}
	// Ensure that there is always at least the root dir.
	if len(wm.wsDirs) == 0 {
		wm.wsDirs = map[span.URI]struct{}{
			wm.root: {},
		}
	}
	wm.built = true
}

// dirs returns the workspace directories for the loaded modules.
func (wm *workspace) dirs(ctx context.Context, fs source.FileSource) []span.URI {
	wm.build(ctx, fs)
	var dirs []span.URI
	for d := range wm.wsDirs {
		dirs = append(dirs, d)
	}
	sort.Slice(dirs, func(i, j int) bool {
		return span.CompareURI(dirs[i], dirs[j]) < 0
	})
	return dirs
}

// invalidate returns a (possibly) new workspaceModule after invalidating
// changedURIs. If wm is still valid in the presence of changedURIs, it returns
// itself unmodified.
func (wm *workspace) invalidate(ctx context.Context, changes map[span.URI]*fileChange) (*workspace, bool) {
	// Prevent races to wm.modFile or wm.wsDirs below, if wm has not yet been
	// built.
	wm.buildMu.Lock()
	defer wm.buildMu.Unlock()
	// Any gopls.mod change is processed first, followed by go.mod changes, as
	// changes to gopls.mod may affect the set of active go.mod files.
	var (
		// New values. We return a new workspace module if and only if modFiles is
		// non-nil.
		modFiles     map[span.URI]struct{}
		moduleSource = wm.moduleSource
		modFile      = wm.file
		err          error
	)
	if wm.moduleSource == goplsModWorkspace {
		// If we are currently reading the modfile from gopls.mod, we default to
		// preserving it even if module metadata changes (which may be the case if
		// a go.sum file changes).
		modFile = wm.file
	}
	// First handle changes to the gopls.mod file.
	if wm.moduleSource != legacyWorkspace {
		// If gopls.mod has changed we need to either re-read it if it exists or
		// walk the filesystem if it doesn't exist.
		gmURI := goplsModURI(wm.root)
		if change, ok := changes[gmURI]; ok {
			if change.exists {
				// Only invalidate if the gopls.mod actually parses. Otherwise, stick with the current gopls.mod
				parsedFile, parsedModules, err := parseGoplsMod(wm.root, gmURI, change.content)
				if err == nil {
					modFile = parsedFile
					moduleSource = goplsModWorkspace
					modFiles = parsedModules
				} else {
					// Note that modFile is not invalidated here.
					event.Error(ctx, "parsing gopls.mod", err)
				}
			} else {
				// gopls.mod is deleted. search for modules again.
				moduleSource = fileSystemWorkspace
				modFiles, err = findAllModules(ctx, wm.root)
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
	if wm.moduleSource != goplsModWorkspace {
		for uri, change := range changes {
			// If a go.mod file has changed, we may need to update the set of active
			// modules.
			if !isGoMod(uri) {
				continue
			}
			if wm.moduleSource == legacyWorkspace && !equalURI(modURI(wm.root), uri) {
				// Legacy mode only considers a module a workspace root.
				continue
			}
			if !source.InDir(wm.root.Filename(), uri.Filename()) {
				// Otherwise, the module must be contained within the workspace root.
				continue
			}
			if modFiles == nil {
				modFiles = make(map[span.URI]struct{})
				for k := range wm.modFiles {
					modFiles[k] = struct{}{}
				}
			}
			if change.exists {
				modFiles[uri] = struct{}{}
			} else {
				delete(modFiles, uri)
			}
		}
	}
	if modFiles != nil {
		// Any change to modules triggers a new version.
		return &workspace{
			root:         wm.root,
			moduleSource: moduleSource,
			modFiles:     modFiles,
			file:         modFile,
			wsDirs:       wm.wsDirs,
		}, true
	}
	// No change. Just return wm, since it is immutable.
	return wm, false
}

func equalURI(left, right span.URI) bool {
	return span.CompareURI(left, right) == 0
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

// findAllModules recursively walks the root directory looking for go.mod
// files, returning the set of modules it discovers.
// TODO(rfindley): consider overlays.
func findAllModules(ctx context.Context, root span.URI) (map[span.URI]struct{}, error) {
	// Walk the view's folder to find all modules in the view.
	modFiles := make(map[span.URI]struct{})
	return modFiles, filepath.Walk(root.Filename(), func(path string, info os.FileInfo, err error) error {
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
		return nil
	})
}

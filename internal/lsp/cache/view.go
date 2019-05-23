// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"context"
	"fmt"
	"go/ast"
	"go/parser"
	"go/types"
	"os"
	"path/filepath"
	"sync"

	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
)

type view struct {
	session *session

	// mu protects all mutable state of the view.
	mu sync.Mutex

	// baseCtx is the context handed to NewView. This is the parent of all
	// background contexts created for this view.
	baseCtx context.Context

	// backgroundCtx is the current context used by background tasks initiated
	// by the view.
	backgroundCtx context.Context

	// cancel is called when all action being performed by the current view
	// should be stopped.
	cancel context.CancelFunc

	// Name is the user visible name of this view.
	name string

	// Folder is the root of this view.
	folder span.URI

	// env is the environment to use when invoking underlying tools.
	env []string

	// keep track of files by uri and by basename, a single file may be mapped
	// to multiple uris, and the same basename may map to multiple files
	filesByURI  map[span.URI]viewFile
	filesByBase map[string][]viewFile

	// contentChanges saves the content changes for a given state of the view.
	// When type information is requested by the view, all of the dirty changes
	// are applied, potentially invalidating some data in the caches. The
	// closures  in the dirty slice assume that their caller is holding the
	// view's mutex.
	contentChanges map[span.URI]func()

	// mcache caches metadata for the packages of the opened files in a view.
	mcache *metadataCache

	// pcache caches type information for the packages of the opened files in a view.
	pcache *packageCache

	// builtinPkg is the AST package used to resolve builtin types.
	builtinPkg *ast.Package

	// ignoredURIs is the set of URIs of files that we ignore.
	ignoredURIs map[span.URI]struct{}
}

type metadataCache struct {
	mu       sync.Mutex
	packages map[string]*metadata
}

type metadata struct {
	id, pkgPath, name string
	files             []string
	typesSizes        types.Sizes
	parents, children map[string]bool
}

type packageCache struct {
	mu       sync.Mutex
	packages map[string]*entry
}

type entry struct {
	pkg   *pkg
	err   error
	ready chan struct{} // closed to broadcast ready condition
}

func (v *view) Session() source.Session {
	return v.session
}

// Name returns the user visible name of this view.
func (v *view) Name() string {
	return v.name
}

// Folder returns the root of this view.
func (v *view) Folder() span.URI {
	return v.folder
}

// Config returns the configuration used for the view's interaction with the
// go/packages API. It is shared across all views.
func (v *view) buildConfig() *packages.Config {
	//TODO:should we cache the config and/or overlay somewhere?
	folderPath, err := v.folder.Filename()
	if err != nil {
		folderPath = ""
	}
	return &packages.Config{
		Context: v.backgroundCtx,
		Dir:     folderPath,
		Env:     v.env,
		Mode: packages.NeedName |
			packages.NeedFiles |
			packages.NeedCompiledGoFiles |
			packages.NeedImports |
			packages.NeedDeps |
			packages.NeedTypesSizes,
		Fset:      v.session.cache.fset,
		Overlay:   v.session.buildOverlay(),
		ParseFile: parseFile,
		Tests:     true,
	}
}

func (v *view) Env() []string {
	v.mu.Lock()
	defer v.mu.Unlock()
	return v.env
}

func (v *view) SetEnv(env []string) {
	v.mu.Lock()
	defer v.mu.Unlock()
	//TODO: this should invalidate the entire view
	v.env = env
}

func (v *view) Shutdown(ctx context.Context) {
	v.session.removeView(ctx, v)
}

func (v *view) shutdown(context.Context) {
	v.mu.Lock()
	defer v.mu.Unlock()
	if v.cancel != nil {
		v.cancel()
		v.cancel = nil
	}
}

// Ignore checks if the given URI is a URI we ignore.
// As of right now, we only ignore files in the "builtin" package.
func (v *view) Ignore(uri span.URI) bool {
	_, ok := v.ignoredURIs[uri]
	return ok
}

func (v *view) BackgroundContext() context.Context {
	v.mu.Lock()
	defer v.mu.Unlock()

	return v.backgroundCtx
}

func (v *view) BuiltinPackage() *ast.Package {
	if v.builtinPkg == nil {
		v.buildBuiltinPkg()
	}
	return v.builtinPkg
}

func (v *view) buildBuiltinPkg() {
	v.mu.Lock()
	defer v.mu.Unlock()
	cfg := *v.buildConfig()
	pkgs, _ := packages.Load(&cfg, "builtin")
	if len(pkgs) != 1 {
		v.builtinPkg, _ = ast.NewPackage(cfg.Fset, nil, nil, nil)
		return
	}
	pkg := pkgs[0]
	files := make(map[string]*ast.File)
	for _, filename := range pkg.GoFiles {
		file, err := parser.ParseFile(cfg.Fset, filename, nil, parser.ParseComments)
		if err != nil {
			v.builtinPkg, _ = ast.NewPackage(cfg.Fset, nil, nil, nil)
			return
		}
		files[filename] = file
		v.ignoredURIs[span.NewURI(filename)] = struct{}{}
	}
	v.builtinPkg, _ = ast.NewPackage(cfg.Fset, files, nil, nil)
}

// SetContent sets the overlay contents for a file.
func (v *view) SetContent(ctx context.Context, uri span.URI, content []byte) error {
	v.mu.Lock()
	defer v.mu.Unlock()

	// Cancel all still-running previous requests, since they would be
	// operating on stale data.
	v.cancel()
	v.backgroundCtx, v.cancel = context.WithCancel(v.baseCtx)

	v.contentChanges[uri] = func() {
		v.session.SetOverlay(uri, content)
	}

	return nil
}

// applyContentChanges applies all of the changed content stored in the view.
// It is assumed that the caller has locked both the view's and the mcache's
// mutexes.
func (v *view) applyContentChanges(ctx context.Context) error {
	if ctx.Err() != nil {
		return ctx.Err()
	}

	v.pcache.mu.Lock()
	defer v.pcache.mu.Unlock()

	for uri, change := range v.contentChanges {
		change()
		delete(v.contentChanges, uri)
	}

	return nil
}

func (f *goFile) invalidate() {
	// TODO(rstambler): Should we recompute these here?
	f.ast = nil
	f.token = nil

	// Remove the package and all of its reverse dependencies from the cache.
	if f.pkg != nil {
		f.view.remove(f.pkg.pkgPath, map[string]struct{}{})
	}
	f.fc = nil
}

// remove invalidates a package and its reverse dependencies in the view's
// package cache. It is assumed that the caller has locked both the mutexes
// of both the mcache and the pcache.
func (v *view) remove(pkgPath string, seen map[string]struct{}) {
	if _, ok := seen[pkgPath]; ok {
		return
	}
	m, ok := v.mcache.packages[pkgPath]
	if !ok {
		return
	}
	seen[pkgPath] = struct{}{}
	for parentPkgPath := range m.parents {
		v.remove(parentPkgPath, seen)
	}
	// All of the files in the package may also be holding a pointer to the
	// invalidated package.
	for _, filename := range m.files {
		if f, _ := v.findFile(span.FileURI(filename)); f != nil {
			if gof, ok := f.(*goFile); ok {
				gof.pkg = nil
			}
		}
	}
	delete(v.pcache.packages, pkgPath)
}

// FindFile returns the file if the given URI is already a part of the view.
func (v *view) FindFile(ctx context.Context, uri span.URI) source.File {
	v.mu.Lock()
	defer v.mu.Unlock()
	f, err := v.findFile(uri)
	if err != nil {
		return nil
	}
	return f
}

// GetFile returns a File for the given URI. It will always succeed because it
// adds the file to the managed set if needed.
func (v *view) GetFile(ctx context.Context, uri span.URI) (source.File, error) {
	v.mu.Lock()
	defer v.mu.Unlock()

	if ctx.Err() != nil {
		return nil, ctx.Err()
	}

	return v.getFile(uri)
}

// getFile is the unlocked internal implementation of GetFile.
func (v *view) getFile(uri span.URI) (viewFile, error) {
	if f, err := v.findFile(uri); err != nil {
		return nil, err
	} else if f != nil {
		return f, nil
	}
	filename, err := uri.Filename()
	if err != nil {
		return nil, err
	}
	var f viewFile
	switch ext := filepath.Ext(filename); ext {
	case ".go":
		f = &goFile{
			fileBase: fileBase{
				view:  v,
				fname: filename,
			},
		}
		v.session.filesWatchMap.Watch(uri, func() {
			f.(*goFile).invalidate()
		})
	case ".mod":
		f = &modFile{
			fileBase: fileBase{
				view:  v,
				fname: filename,
			},
		}
	case ".sum":
		f = &sumFile{
			fileBase: fileBase{
				view:  v,
				fname: filename,
			},
		}
	default:
		return nil, fmt.Errorf("unsupported file extension: %s", ext)
	}
	v.mapFile(uri, f)
	return f, nil
}

// findFile checks the cache for any file matching the given uri.
//
// An error is only returned for an irreparable failure, for example, if the
// filename in question does not exist.
func (v *view) findFile(uri span.URI) (viewFile, error) {
	if f := v.filesByURI[uri]; f != nil {
		// a perfect match
		return f, nil
	}
	// no exact match stored, time to do some real work
	// check for any files with the same basename
	fname, err := uri.Filename()
	if err != nil {
		return nil, err
	}
	basename := basename(fname)
	if candidates := v.filesByBase[basename]; candidates != nil {
		pathStat, err := os.Stat(fname)
		if os.IsNotExist(err) {
			return nil, err
		} else if err != nil {
			return nil, nil // the file may exist, return without an error
		}
		for _, c := range candidates {
			if cStat, err := os.Stat(c.filename()); err == nil {
				if os.SameFile(pathStat, cStat) {
					// same file, map it
					v.mapFile(uri, c)
					return c, nil
				}
			}
		}
	}
	// no file with a matching name was found, it wasn't in our cache
	return nil, nil
}

func (f *fileBase) addURI(uri span.URI) int {
	f.uris = append(f.uris, uri)
	return len(f.uris)
}

func (v *view) mapFile(uri span.URI, f viewFile) {
	v.filesByURI[uri] = f
	if f.addURI(uri) == 1 {
		basename := basename(f.filename())
		v.filesByBase[basename] = append(v.filesByBase[basename], f)
	}
}

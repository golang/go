// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"context"
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"os"
	"path/filepath"
	"sync"

	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/lsp/debug"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
)

type view struct {
	session *session
	id      string

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

	// buildFlags is the build flags to use when invoking underlying tools.
	buildFlags []string

	// keep track of files by uri and by basename, a single file may be mapped
	// to multiple uris, and the same basename may map to multiple files
	filesByURI  map[span.URI]viewFile
	filesByBase map[string][]viewFile

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
	mu       sync.Mutex // guards both maps
	packages map[packageID]*metadata
	ids      map[packagePath]packageID
}

type metadata struct {
	id                packageID
	pkgPath           packagePath
	name              string
	files             []string
	typesSizes        types.Sizes
	parents, children map[packageID]bool
}

type packageCache struct {
	mu       sync.Mutex
	packages map[packageID]*entry
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
func (v *view) Config() *packages.Config {
	// TODO: Should we cache the config and/or overlay somewhere?
	return &packages.Config{
		Dir:        v.folder.Filename(),
		Env:        v.env,
		BuildFlags: v.buildFlags,
		Mode: packages.NeedName |
			packages.NeedFiles |
			packages.NeedCompiledGoFiles |
			packages.NeedImports |
			packages.NeedDeps |
			packages.NeedTypesSizes,
		Fset:    v.session.cache.fset,
		Overlay: v.session.buildOverlay(),
		ParseFile: func(*token.FileSet, string, []byte) (*ast.File, error) {
			panic("go/packages must not be used to parse files")
		},
		Tests: true,
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

func (v *view) SetBuildFlags(buildFlags []string) {
	v.mu.Lock()
	defer v.mu.Unlock()
	v.buildFlags = buildFlags
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
	debug.DropView(debugView{v})
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
	return v.builtinPkg
}

// buildBuiltinPkg builds the view's builtin package.
// It assumes that the view is not active yet,
// i.e. it has not been added to the session's list of views.
func (v *view) buildBuiltinPkg() {
	cfg := *v.Config()
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

	v.session.SetOverlay(uri, content)

	return nil
}

// invalidateContent invalidates the content of a Go file,
// including any position and type information that depends on it.
func (f *goFile) invalidateContent(ctx context.Context) {
	// Mutex acquisition order here is important. It must match the order
	// in loadParseTypecheck to avoid deadlocks.
	f.view.mcache.mu.Lock()
	defer f.view.mcache.mu.Unlock()

	f.view.pcache.mu.Lock()
	defer f.view.pcache.mu.Unlock()

	f.handleMu.Lock()
	defer f.handleMu.Unlock()

	f.invalidateAST(ctx)
	f.handle = nil
}

// invalidateAST invalidates the AST of a Go file,
// including any position and type information that depends on it.
func (f *goFile) invalidateAST(ctx context.Context) {
	f.mu.Lock()
	f.ast = nil
	f.token = nil
	pkgs := f.pkgs
	f.mu.Unlock()

	// Remove the package and all of its reverse dependencies from the cache.
	for id, pkg := range pkgs {
		if pkg != nil {
			f.view.remove(ctx, id, map[packageID]struct{}{})
		}
	}
}

// remove invalidates a package and its reverse dependencies in the view's
// package cache. It is assumed that the caller has locked both the mutexes
// of both the mcache and the pcache.
func (v *view) remove(ctx context.Context, id packageID, seen map[packageID]struct{}) {
	if _, ok := seen[id]; ok {
		return
	}
	m, ok := v.mcache.packages[id]
	if !ok {
		return
	}
	seen[id] = struct{}{}
	for parentID := range m.parents {
		v.remove(ctx, parentID, seen)
	}
	// All of the files in the package may also be holding a pointer to the
	// invalidated package.
	for _, filename := range m.files {
		f, err := v.findFile(span.FileURI(filename))
		if err != nil {
			v.session.log.Errorf(ctx, "cannot find file %s: %v", f.URI(), err)
			continue
		}
		gof, ok := f.(*goFile)
		if !ok {
			v.session.log.Errorf(ctx, "non-Go file %v", f.URI())
			continue
		}
		gof.mu.Lock()
		delete(gof.pkgs, id)
		gof.mu.Unlock()
	}
	delete(v.pcache.packages, id)
	return
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

	return v.getFile(ctx, uri)
}

// getFile is the unlocked internal implementation of GetFile.
func (v *view) getFile(ctx context.Context, uri span.URI) (viewFile, error) {
	if f, err := v.findFile(uri); err != nil {
		return nil, err
	} else if f != nil {
		return f, nil
	}
	filename := uri.Filename()
	var f viewFile
	switch ext := filepath.Ext(filename); ext {
	case ".mod":
		f = &modFile{
			fileBase: fileBase{
				view:  v,
				fname: filename,
				kind:  source.Mod,
			},
		}
	case ".sum":
		f = &sumFile{
			fileBase: fileBase{
				view:  v,
				fname: filename,
				kind:  source.Sum,
			},
		}
	default:
		// Assume that all other files are Go files, regardless of extension.
		f = &goFile{
			fileBase: fileBase{
				view:  v,
				fname: filename,
				kind:  source.Go,
			},
		}
		v.session.filesWatchMap.Watch(uri, func() {
			gof, ok := f.(*goFile)
			if !ok {
				return
			}
			gof.invalidateContent(ctx)
		})
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
	fname := uri.Filename()
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

type debugView struct{ *view }

func (v debugView) ID() string             { return v.id }
func (v debugView) Session() debug.Session { return debugSession{v.session} }

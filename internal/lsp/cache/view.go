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
	"sync"

	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/lsp/xlog"
	"golang.org/x/tools/internal/span"
)

type View struct {
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

	// the logger to use to communicate back with the client
	log xlog.Logger

	// Name is the user visible name of this view.
	Name string

	// Folder is the root of this view.
	Folder span.URI

	// Config is the configuration used for the view's interaction with the
	// go/packages API. It is shared across all views.
	Config packages.Config

	// keep track of files by uri and by basename, a single file may be mapped
	// to multiple uris, and the same basename may map to multiple files
	filesByURI  map[span.URI]*File
	filesByBase map[string][]*File

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
	pkg   *Package
	err   error
	ready chan struct{} // closed to broadcast ready condition
}

func NewView(ctx context.Context, log xlog.Logger, name string, folder span.URI, config *packages.Config) *View {
	backgroundCtx, cancel := context.WithCancel(ctx)
	v := &View{
		baseCtx:        ctx,
		backgroundCtx:  backgroundCtx,
		builtinPkg:     builtinPkg(*config),
		cancel:         cancel,
		log:            log,
		Config:         *config,
		Name:           name,
		Folder:         folder,
		filesByURI:     make(map[span.URI]*File),
		filesByBase:    make(map[string][]*File),
		contentChanges: make(map[span.URI]func()),
		mcache: &metadataCache{
			packages: make(map[string]*metadata),
		},
		pcache: &packageCache{
			packages: make(map[string]*entry),
		},
	}
	return v
}

func (v *View) BackgroundContext() context.Context {
	v.mu.Lock()
	defer v.mu.Unlock()

	return v.backgroundCtx
}

func (v *View) BuiltinPackage() *ast.Package {
	return v.builtinPkg
}

func builtinPkg(cfg packages.Config) *ast.Package {
	var bpkg *ast.Package
	cfg.Mode = packages.LoadFiles
	pkgs, _ := packages.Load(&cfg, "builtin")
	if len(pkgs) != 1 {
		bpkg, _ = ast.NewPackage(cfg.Fset, nil, nil, nil)
		return bpkg
	}
	pkg := pkgs[0]
	files := make(map[string]*ast.File)
	for _, filename := range pkg.GoFiles {
		file, err := parser.ParseFile(cfg.Fset, filename, nil, parser.ParseComments)
		if err != nil {
			bpkg, _ = ast.NewPackage(cfg.Fset, nil, nil, nil)
			return bpkg
		}
		files[filename] = file
	}
	bpkg, _ = ast.NewPackage(cfg.Fset, files, nil, nil)
	return bpkg
}

func (v *View) FileSet() *token.FileSet {
	return v.Config.Fset
}

// SetContent sets the overlay contents for a file.
func (v *View) SetContent(ctx context.Context, uri span.URI, content []byte) error {
	v.mu.Lock()
	defer v.mu.Unlock()

	// Cancel all still-running previous requests, since they would be
	// operating on stale data.
	v.cancel()
	v.backgroundCtx, v.cancel = context.WithCancel(v.baseCtx)

	v.contentChanges[uri] = func() {
		v.applyContentChange(uri, content)
	}

	return nil
}

// applyContentChanges applies all of the changed content stored in the view.
// It is assumed that the caller has locked both the view's and the mcache's
// mutexes.
func (v *View) applyContentChanges(ctx context.Context) error {
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

// setContent applies a content update for a given file. It assumes that the
// caller is holding the view's mutex.
func (v *View) applyContentChange(uri span.URI, content []byte) {
	f, err := v.getFile(uri)
	if err != nil {
		return
	}
	f.content = content

	// TODO(rstambler): Should we recompute these here?
	f.ast = nil
	f.token = nil

	// Remove the package and all of its reverse dependencies from the cache.
	if f.pkg != nil {
		v.remove(f.pkg.pkgPath, map[string]struct{}{})
	}

	switch {
	case f.active && content == nil:
		// The file was active, so we need to forget its content.
		f.active = false
		delete(f.view.Config.Overlay, f.filename)
		f.content = nil
	case content != nil:
		// This is an active overlay, so we update the map.
		f.active = true
		f.view.Config.Overlay[f.filename] = f.content
	}
}

// remove invalidates a package and its reverse dependencies in the view's
// package cache. It is assumed that the caller has locked both the mutexes
// of both the mcache and the pcache.
func (v *View) remove(pkgPath string, seen map[string]struct{}) {
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
			f.pkg = nil
		}
	}
	delete(v.pcache.packages, pkgPath)
}

// FindFile returns the file if the given URI is already a part of the view.
func (v *View) FindFile(ctx context.Context, uri span.URI) *File {
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
func (v *View) GetFile(ctx context.Context, uri span.URI) (source.File, error) {
	v.mu.Lock()
	defer v.mu.Unlock()

	if ctx.Err() != nil {
		return nil, ctx.Err()
	}

	return v.getFile(uri)
}

// getFile is the unlocked internal implementation of GetFile.
func (v *View) getFile(uri span.URI) (*File, error) {
	if f, err := v.findFile(uri); err != nil {
		return nil, err
	} else if f != nil {
		return f, nil
	}
	filename, err := uri.Filename()
	if err != nil {
		return nil, err
	}
	f := &File{
		view:     v,
		filename: filename,
	}
	v.mapFile(uri, f)
	return f, nil
}

// findFile checks the cache for any file matching the given uri.
//
// An error is only returned for an irreparable failure, for example, if the
// filename in question does not exist.
func (v *View) findFile(uri span.URI) (*File, error) {
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
			if cStat, err := os.Stat(c.filename); err == nil {
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

func (v *View) mapFile(uri span.URI, f *File) {
	v.filesByURI[uri] = f
	f.uris = append(f.uris, uri)
	if f.basename == "" {
		f.basename = basename(f.filename)
		v.filesByBase[f.basename] = append(v.filesByBase[f.basename], f)
	}
}

func (v *View) Logger() xlog.Logger {
	return v.log
}

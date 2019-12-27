// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package cache implements the caching layer for gopls.
package cache

import (
	"context"
	"fmt"
	"go/ast"
	"go/token"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"time"

	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/imports"
	"golang.org/x/tools/internal/lsp/debug"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/telemetry/log"
	"golang.org/x/tools/internal/telemetry/tag"
	"golang.org/x/tools/internal/xcontext"
	errors "golang.org/x/xerrors"
)

type view struct {
	session *session
	id      string

	options source.Options

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

	// modfiles are the go.mod files attributed to this view.
	modfiles *modfiles

	// Folder is the root of this view.
	folder span.URI

	// process is the process env for this view.
	// Note: this contains cached module and filesystem state.
	//
	// TODO(suzmue): the state cached in the process env is specific to each view,
	// however, there is state that can be shared between views that is not currently
	// cached, like the module cache.
	processEnv       *imports.ProcessEnv
	cacheRefreshTime time.Time

	// modFileVersions stores the last seen versions of the module files that are used
	// by processEnvs resolver.
	// TODO(suzmue): These versions may not actually be on disk.
	modFileVersions map[string]string

	// keep track of files by uri and by basename, a single file may be mapped
	// to multiple uris, and the same basename may map to multiple files
	filesByURI  map[span.URI]*fileBase
	filesByBase map[string][]*fileBase

	snapshotMu sync.Mutex
	snapshot   *snapshot

	// builtin is used to resolve builtin types.
	builtin *builtinPkg

	// ignoredURIs is the set of URIs of files that we ignore.
	ignoredURIsMu sync.Mutex
	ignoredURIs   map[span.URI]struct{}
}

// modfiles holds the real and temporary go.mod files that are attributed to a view.
type modfiles struct {
	real, temp string
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

func (v *view) Options() source.Options {
	return v.options
}

func minorOptionsChange(a, b source.Options) bool {
	// Check if any of the settings that modify our understanding of files have been changed
	if !reflect.DeepEqual(a.Env, b.Env) {
		return false
	}
	if !reflect.DeepEqual(a.BuildFlags, b.BuildFlags) {
		return false
	}
	// the rest of the options are benign
	return true
}

func (v *view) SetOptions(ctx context.Context, options source.Options) (source.View, error) {
	// no need to rebuild the view if the options were not materially changed
	if minorOptionsChange(v.options, options) {
		v.options = options
		return v, nil
	}
	newView, _, err := v.session.updateView(ctx, v, options)
	return newView, err
}

// Config returns the configuration used for the view's interaction with the
// go/packages API. It is shared across all views.
func (v *view) Config(ctx context.Context) *packages.Config {
	// TODO: Should we cache the config and/or overlay somewhere?

	// We want to run the go commands with the -modfile flag if the version of go
	// that we are using supports it.
	buildFlags := v.options.BuildFlags
	if v.modfiles != nil {
		buildFlags = append(buildFlags, fmt.Sprintf("-modfile=%s", v.modfiles.temp))
	}
	return &packages.Config{
		Dir:        v.folder.Filename(),
		Context:    ctx,
		Env:        v.options.Env,
		BuildFlags: buildFlags,
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
		Logf: func(format string, args ...interface{}) {
			if v.options.VerboseOutput {
				log.Print(ctx, fmt.Sprintf(format, args...))
			}
		},
		Tests: true,
	}
}

func (v *view) RunProcessEnvFunc(ctx context.Context, fn func(*imports.Options) error) error {
	v.mu.Lock()
	defer v.mu.Unlock()
	if v.processEnv == nil {
		var err error
		if v.processEnv, err = v.buildProcessEnv(ctx); err != nil {
			return err
		}
	}

	// Before running the user provided function, clear caches in the resolver.
	if v.modFilesChanged() {
		v.processEnv.GetResolver().(*imports.ModuleResolver).ClearForNewMod()
	}

	// Run the user function.
	opts := &imports.Options{
		// Defaults.
		AllErrors:  true,
		Comments:   true,
		Fragment:   true,
		FormatOnly: false,
		TabIndent:  true,
		TabWidth:   8,
		Env:        v.processEnv,
	}

	if err := fn(opts); err != nil {
		return err
	}
	if v.cacheRefreshTime.IsZero() {
		v.cacheRefreshTime = time.Now()
	}

	// If applicable, store the file versions of the 'go.mod' files that are
	// looked at by the resolver.
	v.storeModFileVersions()

	if time.Since(v.cacheRefreshTime) > 30*time.Second {
		go func() {
			v.mu.Lock()
			defer v.mu.Unlock()

			log.Print(context.Background(), "background imports cache refresh starting")
			v.processEnv.GetResolver().ClearForNewScan()
			// TODO(heschi): prime the cache
			v.cacheRefreshTime = time.Now()
			log.Print(context.Background(), "background refresh finished with err: ", tag.Of("err", nil))
		}()
	}

	return nil
}

func (v *view) buildProcessEnv(ctx context.Context) (*imports.ProcessEnv, error) {
	cfg := v.Config(ctx)
	env := &imports.ProcessEnv{
		WorkingDir: cfg.Dir,
		Logf: func(format string, args ...interface{}) {
			log.Print(ctx, fmt.Sprintf(format, args...))
		},
		LocalPrefix: v.options.LocalPrefix,
		Debug:       v.options.VerboseOutput,
	}
	for _, kv := range cfg.Env {
		split := strings.Split(kv, "=")
		if len(split) < 2 {
			continue
		}
		switch split[0] {
		case "GOPATH":
			env.GOPATH = split[1]
		case "GOROOT":
			env.GOROOT = split[1]
		case "GO111MODULE":
			env.GO111MODULE = split[1]
		case "GOPROXY":
			env.GOPROXY = split[1]
		case "GOFLAGS":
			env.GOFLAGS = split[1]
		case "GOSUMDB":
			env.GOSUMDB = split[1]
		}
	}

	if env.GOPATH == "" {
		cmd := exec.CommandContext(ctx, "go", "env", "GOPATH")
		cmd.Env = cfg.Env
		if out, err := cmd.CombinedOutput(); err != nil {
			return nil, err
		} else {
			env.GOPATH = strings.TrimSpace(string(out))
		}
	}
	return env, nil
}

func (v *view) modFilesChanged() bool {
	// Check the versions of the 'go.mod' files of the main module
	// and modules included by a replace directive. Return true if
	// any of these file versions do not match.
	for filename, version := range v.modFileVersions {
		if version != v.fileVersion(filename, source.Mod) {
			return true
		}
	}
	return false
}

func (v *view) storeModFileVersions() {
	// Store the mod files versions, if we are using a ModuleResolver.
	r, moduleMode := v.processEnv.GetResolver().(*imports.ModuleResolver)
	if !moduleMode || !r.Initialized {
		return
	}
	v.modFileVersions = make(map[string]string)

	// Get the file versions of the 'go.mod' files of the main module
	// and modules included by a replace directive in the resolver.
	for _, mod := range r.ModsByModPath {
		if (mod.Main || mod.Replace != nil) && mod.GoMod != "" {
			v.modFileVersions[mod.GoMod] = v.fileVersion(mod.GoMod, source.Mod)
		}
	}
}

func (v *view) fileVersion(filename string, kind source.FileKind) string {
	uri := span.FileURI(filename)
	fh := v.session.GetFile(uri, kind)
	return fh.Identity().String()
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
	if v.modfiles != nil {
		os.Remove(v.modfiles.temp)
	}
	debug.DropView(debugView{v})
}

// Ignore checks if the given URI is a URI we ignore.
// As of right now, we only ignore files in the "builtin" package.
func (v *view) Ignore(uri span.URI) bool {
	v.ignoredURIsMu.Lock()
	defer v.ignoredURIsMu.Unlock()

	_, ok := v.ignoredURIs[uri]

	// Files with _ prefixes are always ignored.
	if !ok && strings.HasPrefix(filepath.Base(uri.Filename()), "_") {
		v.ignoredURIs[uri] = struct{}{}
		return true
	}

	return ok
}

func (v *view) BackgroundContext() context.Context {
	v.mu.Lock()
	defer v.mu.Unlock()

	return v.backgroundCtx
}

func (v *view) BuiltinPackage() source.BuiltinPackage {
	return v.builtin
}

func (v *view) Snapshot() source.Snapshot {
	return v.getSnapshot()
}

func (v *view) getSnapshot() *snapshot {
	v.snapshotMu.Lock()
	defer v.snapshotMu.Unlock()

	return v.snapshot
}

// invalidateContent invalidates the content of a Go file,
// including any position and type information that depends on it.
// It returns true if we were already tracking the given file, false otherwise.
func (v *view) invalidateContent(ctx context.Context, uri span.URI, kind source.FileKind, action source.FileAction) source.Snapshot {
	// Detach the context so that content invalidation cannot be canceled.
	ctx = xcontext.Detach(ctx)

	// Cancel all still-running previous requests, since they would be
	// operating on stale data.
	switch action {
	case source.Change, source.Close:
		v.cancelBackground()
	}

	// This should be the only time we hold the view's snapshot lock for any period of time.
	v.snapshotMu.Lock()
	defer v.snapshotMu.Unlock()

	v.snapshot = v.snapshot.clone(ctx, uri, kind)
	return v.snapshot
}

func (v *view) cancelBackground() {
	v.mu.Lock()
	defer v.mu.Unlock()

	v.cancel()
	v.backgroundCtx, v.cancel = context.WithCancel(v.baseCtx)
}

// FindFile returns the file if the given URI is already a part of the view.
func (v *view) findFileLocked(ctx context.Context, uri span.URI) *fileBase {
	v.mu.Lock()
	defer v.mu.Unlock()

	f, err := v.findFile(uri)
	if err != nil {
		return nil
	}
	return f
}

// getFileLocked returns a File for the given URI. It will always succeed because it
// adds the file to the managed set if needed.
func (v *view) getFileLocked(ctx context.Context, uri span.URI) (*fileBase, error) {
	v.mu.Lock()
	defer v.mu.Unlock()

	// TODO(rstambler): Should there be a version that provides a kind explicitly?
	kind := source.DetectLanguage("", uri.Filename())
	return v.getFile(ctx, uri, kind)
}

// getFile is the unlocked internal implementation of GetFile.
func (v *view) getFile(ctx context.Context, uri span.URI, kind source.FileKind) (*fileBase, error) {
	f, err := v.findFile(uri)
	if err != nil {
		return nil, err
	} else if f != nil {
		return f, nil
	}
	f = &fileBase{
		view:  v,
		fname: uri.Filename(),
		kind:  kind,
	}
	v.mapFile(uri, f)
	return f, nil
}

// findFile checks the cache for any file matching the given uri.
//
// An error is only returned for an irreparable failure, for example, if the
// filename in question does not exist.
func (v *view) findFile(uri span.URI) (*fileBase, error) {
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
		}
		if err != nil {
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

func (v *view) mapFile(uri span.URI, f *fileBase) {
	v.filesByURI[uri] = f
	if f.addURI(uri) == 1 {
		basename := basename(f.filename())
		v.filesByBase[basename] = append(v.filesByBase[basename], f)
	}
}

func (v *view) FindPosInPackage(searchpkg source.Package, pos token.Pos) (*ast.File, source.Package, error) {
	tok := v.session.cache.fset.File(pos)
	if tok == nil {
		return nil, nil, errors.Errorf("no file for pos in package %s", searchpkg.ID())
	}
	uri := span.FileURI(tok.Name())

	// Special case for ignored files.
	var (
		ph  source.ParseGoHandle
		pkg source.Package
		err error
	)
	if v.Ignore(uri) {
		ph, pkg, err = v.findIgnoredFile(uri)
	} else {
		ph, pkg, err = findFileInPackage(searchpkg, uri)
	}
	if err != nil {
		return nil, nil, err
	}
	file, _, _, err := ph.Cached()
	if err != nil {
		return nil, nil, err
	}
	if !(file.Pos() <= pos && pos <= file.End()) {
		return nil, nil, fmt.Errorf("pos %v, apparently in file %q, is not between %v and %v", pos, ph.File().Identity().URI, file.Pos(), file.End())
	}
	return file, pkg, nil
}

func (v *view) FindMapperInPackage(searchpkg source.Package, uri span.URI) (*protocol.ColumnMapper, error) {
	// Special case for ignored files.
	var (
		ph  source.ParseGoHandle
		err error
	)
	if v.Ignore(uri) {
		ph, _, err = v.findIgnoredFile(uri)
	} else {
		ph, _, err = findFileInPackage(searchpkg, uri)
	}
	if err != nil {
		return nil, err
	}
	_, m, _, err := ph.Cached()
	if err != nil {
		return nil, err
	}
	return m, nil
}

func (v *view) findIgnoredFile(uri span.URI) (source.ParseGoHandle, source.Package, error) {
	// Check the builtin package.
	for _, h := range v.BuiltinPackage().CompiledGoFiles() {
		if h.File().Identity().URI == uri {
			return h, nil, nil
		}
	}
	return nil, nil, errors.Errorf("no ignored file for %s", uri)
}

func findFileInPackage(pkg source.Package, uri span.URI) (source.ParseGoHandle, source.Package, error) {
	queue := []source.Package{pkg}
	seen := make(map[string]bool)

	for len(queue) > 0 {
		pkg := queue[0]
		queue = queue[1:]
		seen[pkg.ID()] = true

		if f, err := pkg.File(uri); err == nil {
			return f, pkg, nil
		}
		for _, dep := range pkg.Imports() {
			if !seen[dep.ID()] {
				queue = append(queue, dep)
			}
		}
	}
	return nil, nil, errors.Errorf("no file for %s in package %s", uri, pkg.ID())
}

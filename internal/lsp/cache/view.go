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
	"strings"
	"sync"

	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/imports"
	"golang.org/x/tools/internal/lsp/debug"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/telemetry/log"
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

	// Folder is the root of this view.
	folder span.URI

	// process is the process env for this view.
	// Note: this contains cached module and filesystem state.
	//
	// TODO(suzmue): the state cached in the process env is specific to each view,
	// however, there is state that can be shared between views that is not currently
	// cached, like the module cache.
	processEnv *imports.ProcessEnv

	// modFileVersions stores the last seen versions of the module files that are used
	// by processEnvs resolver.
	// TODO(suzmue): These versions may not actually be on disk.
	modFileVersions map[string]string

	// keep track of files by uri and by basename, a single file may be mapped
	// to multiple uris, and the same basename may map to multiple files
	filesByURI  map[span.URI]viewFile
	filesByBase map[string][]viewFile

	snapshotMu sync.Mutex
	snapshot   *snapshot

	// builtin is used to resolve builtin types.
	builtin *builtinPkg

	// ignoredURIs is the set of URIs of files that we ignore.
	ignoredURIsMu sync.Mutex
	ignoredURIs   map[span.URI]struct{}
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

func (v *view) SetOptions(options source.Options) {
	v.options = options
}

// Config returns the configuration used for the view's interaction with the
// go/packages API. It is shared across all views.
func (v *view) Config(ctx context.Context) *packages.Config {
	// TODO: Should we cache the config and/or overlay somewhere?
	return &packages.Config{
		Dir:        v.folder.Filename(),
		Context:    ctx,
		Env:        v.options.Env,
		BuildFlags: v.options.BuildFlags,
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
			log.Print(ctx, fmt.Sprintf(format, args...))
		},
		Tests: true,
	}
}

func (v *view) RunProcessEnvFunc(ctx context.Context, fn func(*imports.Options) error, opts *imports.Options) error {
	v.mu.Lock()
	defer v.mu.Unlock()
	if v.processEnv == nil {
		var err error
		if v.processEnv, err = v.buildProcessEnv(ctx); err != nil {
			return err
		}
	}

	// Before running the user provided function, clear caches in the resolver.
	if r, ok := v.processEnv.GetResolver().(*imports.ModuleResolver); ok {
		if v.modFilesChanged() {
			r.ClearForNewMod()
		} else {
			r.ClearForNewScan()
		}
	}

	// Run the user function.
	opts.Env = v.processEnv
	if err := fn(opts); err != nil {
		return err
	}

	// If applicable, store the file versions of the 'go.mod' files that are
	// looked at by the resolver.
	v.storeModFileVersions()

	return nil
}

func (v *view) buildProcessEnv(ctx context.Context) (*imports.ProcessEnv, error) {
	cfg := v.Config(ctx)
	env := &imports.ProcessEnv{
		WorkingDir: cfg.Dir,
		Logf: func(format string, args ...interface{}) {
			log.Print(ctx, fmt.Sprintf(format, args...))
		},
		Debug: true,
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
	f := v.session.GetFile(uri, kind)
	return f.Identity().Version
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
	v.ignoredURIsMu.Lock()
	defer v.ignoredURIsMu.Unlock()

	_, ok := v.ignoredURIs[uri]
	return ok
}

func (v *view) findIgnoredFile(ctx context.Context, uri span.URI) (source.ParseGoHandle, source.Package, error) {
	// Check the builtin package.
	for _, h := range v.BuiltinPackage().Files() {
		if h.File().Identity().URI == uri {
			return h, nil, nil
		}
	}
	return nil, nil, errors.Errorf("no ignored file for %s", uri)
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

// SetContent sets the overlay contents for a file.
func (v *view) SetContent(ctx context.Context, uri span.URI, content []byte) (bool, error) {
	v.mu.Lock()
	defer v.mu.Unlock()

	// Cancel all still-running previous requests, since they would be
	// operating on stale data.
	v.cancel()
	v.backgroundCtx, v.cancel = context.WithCancel(v.baseCtx)

	if v.Ignore(uri) {
		return false, nil
	}

	kind := source.DetectLanguage("", uri.Filename())
	return v.session.SetOverlay(uri, kind, content), nil
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

	// TODO(rstambler): Should there be a version that provides a kind explicitly?
	kind := source.DetectLanguage("", uri.Filename())
	return v.getFile(ctx, uri, kind)
}

// getFile is the unlocked internal implementation of GetFile.
func (v *view) getFile(ctx context.Context, uri span.URI, kind source.FileKind) (viewFile, error) {
	f, err := v.findFile(uri)
	if err != nil {
		return nil, err
	} else if f != nil {
		return f, nil
	}
	f = &fileBase{
		view:  v,
		fname: uri.Filename(),
		kind:  source.Go,
	}
	v.session.filesWatchMap.Watch(uri, func(changeType protocol.FileChangeType) bool {
		ctx := xcontext.Detach(ctx)
		return v.invalidateContent(ctx, f, kind, changeType)
	})
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

func (v *view) mapFile(uri span.URI, f viewFile) {
	v.filesByURI[uri] = f
	if f.addURI(uri) == 1 {
		basename := basename(f.filename())
		v.filesByBase[basename] = append(v.filesByBase[basename], f)
	}
}

func (v *view) openFiles(ctx context.Context, uris []span.URI) (results []source.File) {
	v.mu.Lock()
	defer v.mu.Unlock()

	for _, uri := range uris {
		// Call unlocked version of getFile since we hold the lock on the view.
		if f, err := v.getFile(ctx, uri, source.Go); err == nil && v.session.IsOpen(uri) {
			results = append(results, f)
		}
	}
	return results
}

type debugView struct{ *view }

func (v debugView) ID() string             { return v.id }
func (v debugView) Session() debug.Session { return debugSession{v.session} }

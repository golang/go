// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package cache implements the caching layer for gopls.
package cache

import (
	"context"
	"encoding/json"
	"fmt"
	"go/ast"
	"go/token"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"strings"
	"sync"
	"time"

	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/imports"
	"golang.org/x/tools/internal/lsp/debug"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/lsp/telemetry"
	"golang.org/x/tools/internal/memoize"
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

	// mu protects most mutable state of the view.
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

	// importsMu guards imports-related state, particularly the ProcessEnv.
	importsMu sync.Mutex
	// process is the process env for this view.
	// Note: this contains cached module and filesystem state.
	//
	// TODO(suzmue): the state cached in the process env is specific to each view,
	// however, there is state that can be shared between views that is not currently
	// cached, like the module cache.
	processEnv           *imports.ProcessEnv
	cacheRefreshDuration time.Duration
	cacheRefreshTimer    *time.Timer
	cachedModFileVersion source.FileIdentity

	// keep track of files by uri and by basename, a single file may be mapped
	// to multiple uris, and the same basename may map to multiple files
	filesByURI  map[span.URI]*fileBase
	filesByBase map[string][]*fileBase

	snapshotMu sync.Mutex
	snapshot   *snapshot

	// ignoredURIs is the set of URIs of files that we ignore.
	ignoredURIsMu sync.Mutex
	ignoredURIs   map[span.URI]struct{}

	// initialized is closed when the view has been fully initialized.
	// On initialization, the view's workspace packages are loaded.
	// All of the fields below are set as part of initialization.
	// If we failed to load, we don't re-try to avoid too many go/packages calls.
	initializeOnce sync.Once
	initialized    chan struct{}

	// builtin pins the AST and package for builtin.go in memory.
	builtin *builtinPackageHandle

	// True if the view is either in GOPATH, a module, or some other
	// non go command build system.
	hasValidBuildConfiguration bool

	// The real and temporary go.mod files that are attributed to a view.
	// The temporary go.mod is for use with the Go command's -modfile flag.
	realMod, tempMod span.URI

	// goCommand indicates if the user is using the go command or some other
	// build system.
	goCommand bool

	// `go env` variables that need to be tracked.
	gopath, gocache string

	// LoadMu guards packages.Load calls and associated state.
	loadMu         sync.Mutex
	serializeLoads int
}

type builtinPackageHandle struct {
	handle *memoize.Handle
	file   source.ParseGoHandle
}

type builtinPackageData struct {
	memoize.NoCopy

	pkg *ast.Package
	err error
}

// fileBase holds the common functionality for all files.
// It is intended to be embedded in the file implementations
type fileBase struct {
	uris  []span.URI
	fname string

	view *view
}

func (f *fileBase) URI() span.URI {
	return f.uris[0]
}

func (f *fileBase) filename() string {
	return f.fname
}

func (f *fileBase) addURI(uri span.URI) int {
	f.uris = append(f.uris, uri)
	return len(f.uris)
}

func (v *view) ValidBuildConfiguration() bool {
	return v.hasValidBuildConfiguration
}

func (v *view) ModFiles() (span.URI, span.URI) {
	return v.realMod, v.tempMod
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

func (v *view) Rebuild(ctx context.Context) (source.Snapshot, error) {
	_, snapshot, err := v.session.updateView(ctx, v, v.options)
	return snapshot, err
}

func (v *view) LookupBuiltin(ctx context.Context, name string) (*ast.Object, error) {
	v.awaitInitialized(ctx)

	if v.builtin == nil {
		return nil, errors.Errorf("no builtin package for view %s", v.name)
	}
	data := v.builtin.handle.Get(ctx)
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}
	if data == nil {
		return nil, errors.Errorf("unexpected nil builtin package")
	}
	d, ok := data.(*builtinPackageData)
	if !ok {
		return nil, errors.Errorf("unexpected type %T", data)
	}
	if d.err != nil {
		return nil, d.err
	}
	if d.pkg == nil || d.pkg.Scope == nil {
		return nil, errors.Errorf("no builtin package")
	}
	astObj := d.pkg.Scope.Lookup(name)
	if astObj == nil {
		return nil, errors.Errorf("no builtin object for %s", name)
	}
	return astObj, nil
}

func (v *view) buildBuiltinPackage(ctx context.Context, m *metadata) error {
	if len(m.goFiles) != 1 {
		return errors.Errorf("only expected 1 file, got %v", len(m.goFiles))
	}
	uri := m.goFiles[0]
	v.addIgnoredFile(uri) // to avoid showing diagnostics for builtin.go

	// Get the FileHandle through the session to avoid adding it to the snapshot.
	pgh := v.session.cache.ParseGoHandle(v.session.GetFile(uri), source.ParseFull)
	fset := v.session.cache.fset
	h := v.session.cache.store.Bind(pgh.File().Identity(), func(ctx context.Context) interface{} {
		data := &builtinPackageData{}
		file, _, _, err := pgh.Parse(ctx)
		if err != nil {
			data.err = err
			return data
		}
		data.pkg, data.err = ast.NewPackage(fset, map[string]*ast.File{
			pgh.File().Identity().URI.Filename(): file,
		}, nil, nil)
		return data
	})
	v.builtin = &builtinPackageHandle{
		handle: h,
		file:   pgh,
	}
	return nil
}

// Config returns the configuration used for the view's interaction with the
// go/packages API. It is shared across all views.
func (v *view) Config(ctx context.Context) *packages.Config {
	// TODO: Should we cache the config and/or overlay somewhere?

	// We want to run the go commands with the -modfile flag if the version of go
	// that we are using supports it.
	buildFlags := v.options.BuildFlags
	if v.tempMod != "" {
		buildFlags = append(buildFlags, fmt.Sprintf("-modfile=%s", v.tempMod.Filename()))
	}
	cfg := &packages.Config{
		Dir:        v.folder.Filename(),
		Context:    ctx,
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
	cfg.Env = append(cfg.Env, fmt.Sprintf("GOPATH=%s", v.gopath))
	cfg.Env = append(cfg.Env, v.options.Env...)
	return cfg
}

func (v *view) RunProcessEnvFunc(ctx context.Context, fn func(*imports.Options) error) error {
	v.importsMu.Lock()
	defer v.importsMu.Unlock()

	if v.processEnv == nil {
		var err error
		if v.processEnv, err = v.buildProcessEnv(ctx); err != nil {
			return err
		}
	}

	// In module mode, check if the mod file has changed.
	if v.realMod != "" {
		mod, err := v.Snapshot().GetFile(v.realMod)
		if err == nil && mod.Identity() != v.cachedModFileVersion {
			v.processEnv.GetResolver().(*imports.ModuleResolver).ClearForNewMod()
			v.cachedModFileVersion = mod.Identity()
		}
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

	if v.cacheRefreshTimer == nil {
		// Don't refresh more than twice per minute.
		delay := 30 * time.Second
		// Don't spend more than a couple percent of the time refreshing.
		if adaptive := 50 * v.cacheRefreshDuration; adaptive > delay {
			delay = adaptive
		}
		v.cacheRefreshTimer = time.AfterFunc(delay, v.refreshProcessEnv)
	}

	return nil
}

func (v *view) refreshProcessEnv() {
	start := time.Now()

	v.importsMu.Lock()
	env := v.processEnv
	env.GetResolver().ClearForNewScan()
	v.importsMu.Unlock()

	// We don't have a context handy to use for logging, so use the stdlib for now.
	log.Print(v.baseCtx, "background imports cache refresh starting")
	err := imports.PrimeCache(context.Background(), env)
	log.Print(v.baseCtx, fmt.Sprintf("background refresh finished after %v", time.Since(start)), tag.Of("Error", err))

	v.importsMu.Lock()
	v.cacheRefreshDuration = time.Since(start)
	v.cacheRefreshTimer = nil
	v.importsMu.Unlock()
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
	if len(cfg.BuildFlags) > 0 {
		if env.GOFLAGS != "" {
			env.GOFLAGS += " "
		}
		env.GOFLAGS += strings.Join(cfg.BuildFlags, " ")
	}
	return env, nil
}

func (v *view) mapFile(uri span.URI, f *fileBase) {
	v.filesByURI[uri] = f
	if f.addURI(uri) == 1 {
		basename := basename(f.filename())
		v.filesByBase[basename] = append(v.filesByBase[basename], f)
	}
}

func basename(filename string) string {
	return strings.ToLower(filepath.Base(filename))
}

// knownFile returns true if the given URI is already a part of the view.
func (v *view) knownFile(uri span.URI) bool {
	v.mu.Lock()
	defer v.mu.Unlock()

	f, err := v.findFile(uri)
	return f != nil && err == nil
}

// getFile returns a file for the given URI. It will always succeed because it
// adds the file to the managed set if needed.
func (v *view) getFile(uri span.URI) (*fileBase, error) {
	v.mu.Lock()
	defer v.mu.Unlock()

	f, err := v.findFile(uri)
	if err != nil {
		return nil, err
	} else if f != nil {
		return f, nil
	}
	f = &fileBase{
		view:  v,
		fname: uri.Filename(),
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

func (v *view) Shutdown(ctx context.Context) {
	v.session.removeView(ctx, v)
}

func (v *view) shutdown(context.Context) {
	// TODO: Cancel the view's initialization.
	v.mu.Lock()
	defer v.mu.Unlock()
	if v.cancel != nil {
		v.cancel()
		v.cancel = nil
	}
	if v.tempMod != "" {
		os.Remove(v.tempMod.Filename())
		os.Remove(tempSumFile(v.tempMod.Filename()))
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

func (v *view) addIgnoredFile(uri span.URI) {
	v.ignoredURIsMu.Lock()
	defer v.ignoredURIsMu.Unlock()

	v.ignoredURIs[uri] = struct{}{}
}

func (v *view) BackgroundContext() context.Context {
	v.mu.Lock()
	defer v.mu.Unlock()

	return v.backgroundCtx
}

func (v *view) Snapshot() source.Snapshot {
	return v.getSnapshot()
}

func (v *view) getSnapshot() *snapshot {
	v.snapshotMu.Lock()
	defer v.snapshotMu.Unlock()

	return v.snapshot
}

func (v *view) initialize(ctx context.Context, s *snapshot) {
	v.initializeOnce.Do(func() {
		defer close(v.initialized)

		err := func() error {
			// Do not cancel the call to go/packages.Load for the entire workspace.
			meta, err := s.load(ctx, viewLoadScope("LOAD_VIEW"), packagePath("builtin"))
			if err != nil {
				return err
			}
			// Find the builtin package in order to handle it separately.
			for _, m := range meta {
				if m.pkgPath == "builtin" {
					return s.view.buildBuiltinPackage(ctx, m)
				}
			}
			return errors.Errorf("failed to load the builtin package")
		}()
		if err != nil {
			log.Error(ctx, "initial workspace load failed", err)
		}
	})
}

func (v *view) awaitInitialized(ctx context.Context) {
	select {
	case <-ctx.Done():
	case <-v.initialized:
	}
}

// invalidateContent invalidates the content of a Go file,
// including any position and type information that depends on it.
// It returns true if we were already tracking the given file, false otherwise.
func (v *view) invalidateContent(ctx context.Context, uris []span.URI) source.Snapshot {
	// Detach the context so that content invalidation cannot be canceled.
	ctx = xcontext.Detach(ctx)

	// Cancel all still-running previous requests, since they would be
	// operating on stale data.
	v.cancelBackground()

	// Do not clone a snapshot until its view has finished initializing.
	v.awaitInitialized(ctx)

	// This should be the only time we hold the view's snapshot lock for any period of time.
	v.snapshotMu.Lock()
	defer v.snapshotMu.Unlock()

	v.snapshot = v.snapshot.clone(ctx, uris)
	return v.snapshot
}

func (v *view) cancelBackground() {
	v.mu.Lock()
	defer v.mu.Unlock()

	v.cancel()
	v.backgroundCtx, v.cancel = context.WithCancel(v.baseCtx)
}

func (v *view) setBuildInformation(ctx context.Context, folder span.URI, env []string, modfileFlagEnabled bool) error {
	// Make sure to get the `go env` before continuing with initialization.
	gomod, err := v.getGoEnv(ctx, env)
	if err != nil {
		return err
	}
	modFile := strings.TrimSpace(gomod)
	if modFile == os.DevNull {
		return nil
	}
	v.realMod = span.FileURI(modFile)

	// Now that we have set all required fields,
	// check if the view has a valid build configuration.
	v.hasValidBuildConfiguration = checkBuildConfiguration(v.goCommand, v.realMod, v.folder, v.gopath)

	// The user has disabled the use of the -modfile flag.
	if !modfileFlagEnabled {
		return nil
	}
	if modfileFlag, err := v.modfileFlagExists(ctx, v.Options().Env); err != nil {
		return err
	} else if !modfileFlag {
		return nil
	}
	// Copy the current go.mod file into the temporary go.mod file.
	// The file's name will be of the format go.1234.mod.
	// It's temporary go.sum file should have the corresponding format of go.1234.sum.
	tempModFile, err := ioutil.TempFile("", "go.*.mod")
	if err != nil {
		return err
	}
	defer tempModFile.Close()

	origFile, err := os.Open(modFile)
	if err != nil {
		return err
	}
	defer origFile.Close()

	if _, err := io.Copy(tempModFile, origFile); err != nil {
		return err
	}
	v.tempMod = span.FileURI(tempModFile.Name())

	// Copy go.sum file as well (if there is one).
	sumFile := filepath.Join(filepath.Dir(modFile), "go.sum")
	stat, err := os.Stat(sumFile)
	if err != nil || !stat.Mode().IsRegular() {
		return nil
	}
	contents, err := ioutil.ReadFile(sumFile)
	if err != nil {
		return err
	}
	if err := ioutil.WriteFile(tempSumFile(tempModFile.Name()), contents, stat.Mode()); err != nil {
		return err
	}
	return nil
}

func checkBuildConfiguration(goCommand bool, mod, folder span.URI, gopath string) bool {
	// Since we only really understand the `go` command, if the user is not
	// using the go command, assume that their configuration is valid.
	if !goCommand {
		return true
	}
	// Check if the user is working within a module.
	if mod != "" {
		return true
	}
	// The user may have a multiple directories in their GOPATH.
	// Check if the workspace is within any of them.
	for _, gp := range filepath.SplitList(gopath) {
		if isSubdirectory(filepath.Join(gp, "src"), folder.Filename()) {
			return true
		}
	}
	return false
}

func isSubdirectory(root, leaf string) bool {
	rel, err := filepath.Rel(root, leaf)
	return err == nil && !strings.HasPrefix(rel, "..")
}

// getGoEnv sets the view's build information's GOPATH, GOCACHE, and GOPACKAGESDRIVER values.
// It also returns the view's GOMOD value, which need not be cached.
func (v *view) getGoEnv(ctx context.Context, env []string) (string, error) {
	var gocache, gopath, gopackagesdriver bool
	isGoCommand := func(gopackagesdriver string) bool {
		return gopackagesdriver == "" || gopackagesdriver == "off"
	}
	for _, e := range env {
		split := strings.Split(e, "=")
		if len(split) != 2 {
			continue
		}
		switch split[0] {
		case "GOCACHE":
			v.gocache = split[1]
			gocache = true
		case "GOPATH":
			v.gopath = split[1]
			gopath = true
		case "GOPACKAGESDRIVER":
			v.goCommand = isGoCommand(split[1])
			gopackagesdriver = true
		}
	}
	b, err := source.InvokeGo(ctx, v.folder.Filename(), env, "env", "-json")
	if err != nil {
		return "", err
	}
	envMap := make(map[string]string)
	decoder := json.NewDecoder(b)
	if err := decoder.Decode(&envMap); err != nil {
		return "", err
	}
	if !gopath {
		if gopath, ok := envMap["GOPATH"]; ok {
			v.gopath = gopath
		} else {
			return "", errors.New("unable to determine GOPATH")
		}
	}
	if !gocache {
		if gocache, ok := envMap["GOCACHE"]; ok {
			v.gocache = gocache
		} else {
			return "", errors.New("unable to determine GOCACHE")
		}
	}
	// The value of GOPACKAGESDRIVER is not returned through the go command.
	if !gopackagesdriver {
		v.goCommand = isGoCommand(os.Getenv("GOPACKAGESDRIVER"))
	}
	if gomod, ok := envMap["GOMOD"]; ok {
		return gomod, nil
	}
	return "", nil
}

// 1.13: go: updates to go.mod needed, but contents have changed
// 1.14: go: updating go.mod: existing contents have changed since last read
var modConcurrencyError = regexp.MustCompile(`go:.*go.mod.*contents have changed`)

// LoadPackages calls packages.Load, serializing requests if they fight over
// go.mod changes.
func (v *view) loadPackages(cfg *packages.Config, patterns ...string) ([]*packages.Package, error) {
	// We want to run go list calls concurrently as much as possible. However,
	// if go.mod updates are needed, only one can make them and the others will
	// fail. We need to retry in those cases, but we don't want to thrash so
	// badly we never recover. To avoid that, once we've seen one concurrency
	// error, start serializing everything until the backlog has cleared out.
	// This could all be avoided on 1.14 by using multiple -modfiles.

	v.loadMu.Lock()
	var locked bool // If true, we hold the mutex and have incremented.
	if v.serializeLoads == 0 {
		v.loadMu.Unlock()
	} else {
		locked = true
		v.serializeLoads++
	}
	defer func() {
		if locked {
			v.serializeLoads--
			v.loadMu.Unlock()
		}
	}()

	for {
		pkgs, err := packages.Load(cfg, patterns...)
		if err == nil || !modConcurrencyError.MatchString(err.Error()) {
			return pkgs, err
		}

		log.Error(cfg.Context, "Load concurrency error, will retry serially", err)
		if !locked {
			v.loadMu.Lock()
			v.serializeLoads++
			locked = true
		}
	}
}

// This function will return the main go.mod file for this folder if it exists and whether the -modfile
// flag exists for this version of go.
func (v *view) modfileFlagExists(ctx context.Context, env []string) (bool, error) {
	// Check the go version by running "go list" with modules off.
	// Borrowed from internal/imports/mod.go:620.
	const format = `{{range context.ReleaseTags}}{{if eq . "go1.14"}}{{.}}{{end}}{{end}}`
	folder := v.folder.Filename()
	stdout, err := source.InvokeGo(ctx, folder, append(env, "GO111MODULE=off"), "list", "-e", "-f", format)
	if err != nil {
		return false, err
	}
	// If the output is not go1.14 or an empty string, then it could be an error.
	lines := strings.Split(stdout.String(), "\n")
	if len(lines) < 2 && stdout.String() != "" {
		log.Error(ctx, "unexpected stdout when checking for go1.14", errors.Errorf("%q", stdout), telemetry.Directory.Of(folder))
		return false, nil
	}
	return lines[0] == "go1.14", nil
}

// tempSumFile returns the path to the copied temporary go.sum file.
// It simply replaces the extension of the temporary go.mod file with "sum".
func tempSumFile(filename string) string {
	if filename == "" {
		return ""
	}
	return filename[:len(filename)-len("mod")] + "sum"
}

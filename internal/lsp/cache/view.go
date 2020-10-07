// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package cache implements the caching layer for gopls.
package cache

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"reflect"
	"regexp"
	"sort"
	"strings"
	"sync"

	"golang.org/x/mod/modfile"
	"golang.org/x/mod/semver"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/gocommand"
	"golang.org/x/tools/internal/imports"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/memoize"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/xcontext"
	errors "golang.org/x/xerrors"
)

type View struct {
	session *Session
	id      string

	optionsMu sync.Mutex
	options   *source.Options

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

	// name is the user visible name of this view.
	name string

	// folder is the folder with which this view was constructed.
	folder span.URI

	importsState *importsState

	// keep track of files by uri and by basename, a single file may be mapped
	// to multiple uris, and the same basename may map to multiple files
	filesByURI  map[span.URI]*fileBase
	filesByBase map[string][]*fileBase

	snapshotMu sync.Mutex
	snapshot   *snapshot

	// initialized is closed when the view has been fully initialized. On
	// initialization, the view's workspace packages are loaded. All of the
	// fields below are set as part of initialization. If we failed to load, we
	// only retry if the go.mod file changes, to avoid too many go/packages
	// calls.
	//
	// When the view is created, initializeOnce is non-nil, initialized is
	// open, and initCancelFirstAttempt can be used to terminate
	// initialization. Once initialization completes, initializedErr may be set
	// and initializeOnce becomes nil. If initializedErr is non-nil,
	// initialization may be retried (depending on how files are changed). To
	// indicate that initialization should be retried, initializeOnce will be
	// set. The next time a caller requests workspace packages, the
	// initialization will retry.
	initialized            chan struct{}
	initCancelFirstAttempt context.CancelFunc

	// initializationSema is used as a mutex to guard initializeOnce and
	// initializedErr, which will be updated after each attempt to initialize
	// the view. We use a channel instead of a mutex to avoid blocking when a
	// context is canceled.
	initializationSema chan struct{}
	initializeOnce     *sync.Once
	initializedErr     error

	// workspaceInformation tracks various details about this view's
	// environment variables, go version, and use of modules.
	workspaceInformation
}

type workspaceInformation struct {
	// The Go version in use: X in Go 1.X.
	goversion int

	// hasGopackagesDriver is true if the user has a value set for the
	// GOPACKAGESDRIVER environment variable or a gopackagesdriver binary on
	// their machine.
	hasGopackagesDriver bool

	// `go env` variables that need to be tracked by gopls.
	environmentVariables

	// The value of GO111MODULE we want to run with.
	go111module string

	// goEnv is the `go env` output collected when a view is created.
	// It includes the values of the environment variables above.
	goEnv map[string]string

	// rootURI is the rootURI directory of this view. If we are in GOPATH mode, this
	// is just the folder. If we are in module mode, this is the module rootURI.
	rootURI span.URI
}

type environmentVariables struct {
	gocache, gopath, goprivate, gomodcache, gomod string
}

type workspaceMode int

const (
	moduleMode workspaceMode = 1 << iota

	// tempModfile indicates whether or not the -modfile flag should be used.
	tempModfile

	// usesWorkspaceModule indicates support for the experimental workspace module
	// feature.
	usesWorkspaceModule
)

type builtinPackageHandle struct {
	handle *memoize.Handle
}

type builtinPackageData struct {
	parsed *source.BuiltinPackage
	err    error
}

type moduleRoot struct {
	rootURI        span.URI
	modURI, sumURI span.URI
}

// fileBase holds the common functionality for all files.
// It is intended to be embedded in the file implementations
type fileBase struct {
	uris  []span.URI
	fname string

	view *View
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

func (v *View) ID() string { return v.id }

// tempModFile creates a temporary go.mod file based on the contents of the
// given go.mod file. It is the caller's responsibility to clean up the files
// when they are done using them.
func tempModFile(modFh, sumFH source.FileHandle) (tmpURI span.URI, cleanup func(), err error) {
	filenameHash := hashContents([]byte(modFh.URI().Filename()))
	tmpMod, err := ioutil.TempFile("", fmt.Sprintf("go.%s.*.mod", filenameHash))
	if err != nil {
		return "", nil, err
	}
	defer tmpMod.Close()

	tmpURI = span.URIFromPath(tmpMod.Name())
	tmpSumName := sumFilename(tmpURI)

	content, err := modFh.Read()
	if err != nil {
		return "", nil, err
	}

	if _, err := tmpMod.Write(content); err != nil {
		return "", nil, err
	}

	cleanup = func() {
		_ = os.Remove(tmpSumName)
		_ = os.Remove(tmpURI.Filename())
	}

	// Be careful to clean up if we return an error from this function.
	defer func() {
		if err != nil {
			cleanup()
			cleanup = nil
		}
	}()

	// Create an analogous go.sum, if one exists.
	if sumFH != nil {
		sumContents, err := sumFH.Read()
		if err != nil {
			return "", cleanup, err
		}
		if err := ioutil.WriteFile(tmpSumName, sumContents, 0655); err != nil {
			return "", cleanup, err
		}
	}

	return tmpURI, cleanup, nil
}

// Name returns the user visible name of this view.
func (v *View) Name() string {
	return v.name
}

// Folder returns the folder at the base of this view.
func (v *View) Folder() span.URI {
	return v.folder
}

func (v *View) Options() *source.Options {
	v.optionsMu.Lock()
	defer v.optionsMu.Unlock()
	return v.options
}

func minorOptionsChange(a, b *source.Options) bool {
	// Check if any of the settings that modify our understanding of files have been changed
	mapEnv := func(env []string) map[string]string {
		m := make(map[string]string, len(env))
		for _, x := range env {
			split := strings.SplitN(x, "=", 2)
			if len(split) != 2 {
				continue
			}
			m[split[0]] = split[1]
		}
		return m
	}
	aEnv := mapEnv(a.Env)
	bEnv := mapEnv(b.Env)
	if !reflect.DeepEqual(aEnv, bEnv) {
		return false
	}
	aBuildFlags := make([]string, len(a.BuildFlags))
	bBuildFlags := make([]string, len(b.BuildFlags))
	copy(aBuildFlags, a.BuildFlags)
	copy(bBuildFlags, b.BuildFlags)
	sort.Strings(aBuildFlags)
	sort.Strings(bBuildFlags)
	// the rest of the options are benign
	return reflect.DeepEqual(aBuildFlags, bBuildFlags)
}

func (v *View) SetOptions(ctx context.Context, options *source.Options) (source.View, error) {
	// no need to rebuild the view if the options were not materially changed
	v.optionsMu.Lock()
	if minorOptionsChange(v.options, options) {
		v.options = options
		v.optionsMu.Unlock()
		return v, nil
	}
	v.optionsMu.Unlock()
	newView, err := v.session.updateView(ctx, v, options)
	return newView, err
}

func (v *View) Rebuild(ctx context.Context) (source.Snapshot, func(), error) {
	newView, err := v.session.updateView(ctx, v, v.Options())
	if err != nil {
		return nil, func() {}, err
	}
	snapshot, release := newView.Snapshot(ctx)
	return snapshot, release, nil
}

func (s *snapshot) WriteEnv(ctx context.Context, w io.Writer) error {
	s.view.optionsMu.Lock()
	env, buildFlags := s.view.envLocked()
	s.view.optionsMu.Unlock()

	fullEnv := make(map[string]string)
	for k, v := range s.view.goEnv {
		fullEnv[k] = v
	}
	for _, v := range env {
		s := strings.SplitN(v, "=", 2)
		if len(s) != 2 {
			continue
		}
		if _, ok := fullEnv[s[0]]; ok {
			fullEnv[s[0]] = s[1]
		}
	}
	goVersion, err := s.view.session.gocmdRunner.Run(ctx, gocommand.Invocation{
		Verb:       "version",
		Env:        env,
		WorkingDir: s.view.rootURI.Filename(),
	})
	if err != nil {
		return err
	}
	fmt.Fprintf(w, `go env for %v
(root %s)
(go version %s)
(valid build configuration = %v)
(build flags: %v)
`,
		s.view.folder.Filename(),
		s.view.rootURI.Filename(),
		strings.TrimRight(goVersion.String(), "\n"),
		s.ValidBuildConfiguration(),
		buildFlags)
	for k, v := range fullEnv {
		fmt.Fprintf(w, "%s=%s\n", k, v)
	}
	return nil
}

func (s *snapshot) RunProcessEnvFunc(ctx context.Context, fn func(*imports.Options) error) error {
	return s.view.importsState.runProcessEnvFunc(ctx, s, fn)
}

// envLocked returns the environment and build flags for the current view.
// It assumes that the caller is holding the view's optionsMu.
func (v *View) envLocked() ([]string, []string) {
	env := append(os.Environ(), v.options.Env...)
	buildFlags := append([]string{}, v.options.BuildFlags...)
	return env, buildFlags
}

func (v *View) contains(uri span.URI) bool {
	return strings.HasPrefix(string(uri), string(v.rootURI))
}

func (v *View) mapFile(uri span.URI, f *fileBase) {
	v.filesByURI[uri] = f
	if f.addURI(uri) == 1 {
		basename := basename(f.filename())
		v.filesByBase[basename] = append(v.filesByBase[basename], f)
	}
}

func basename(filename string) string {
	return strings.ToLower(filepath.Base(filename))
}

func (v *View) relevantChange(c source.FileModification) bool {
	// If the file is known to the view, the change is relevant.
	known := v.knownFile(c.URI)

	// If the file is not known to the view, and the change is only on-disk,
	// we should not invalidate the snapshot. This is necessary because Emacs
	// sends didChangeWatchedFiles events for temp files.
	if !known && c.OnDisk && (c.Action == source.Change || c.Action == source.Delete) {
		return false
	}
	return v.contains(c.URI) || known
}

func (v *View) knownFile(uri span.URI) bool {
	v.mu.Lock()
	defer v.mu.Unlock()

	f, err := v.findFile(uri)
	return f != nil && err == nil
}

// getFile returns a file for the given URI. It will always succeed because it
// adds the file to the managed set if needed.
func (v *View) getFile(uri span.URI) (*fileBase, error) {
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
func (v *View) findFile(uri span.URI) (*fileBase, error) {
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

func (v *View) Shutdown(ctx context.Context) {
	v.session.removeView(ctx, v)
}

func (v *View) shutdown(ctx context.Context) {
	// Cancel the initial workspace load if it is still running.
	v.initCancelFirstAttempt()

	v.mu.Lock()
	if v.cancel != nil {
		v.cancel()
		v.cancel = nil
	}
	v.mu.Unlock()
	v.snapshotMu.Lock()
	go v.snapshot.generation.Destroy()
	v.snapshotMu.Unlock()
}

func (v *View) BackgroundContext() context.Context {
	v.mu.Lock()
	defer v.mu.Unlock()

	return v.backgroundCtx
}

func (s *snapshot) IgnoredFile(uri span.URI) bool {
	filename := uri.Filename()
	var prefixes []string
	if len(s.modules) == 0 {
		for _, entry := range filepath.SplitList(s.view.gopath) {
			prefixes = append(prefixes, filepath.Join(entry, "src"))
		}
	} else {
		prefixes = append(prefixes, s.view.gomodcache)
		for _, m := range s.modules {
			prefixes = append(prefixes, m.rootURI.Filename())
		}
	}
	for _, prefix := range prefixes {
		if strings.HasPrefix(filename, prefix) {
			return checkIgnored(filename[len(prefix):])
		}
	}
	return false
}

// checkIgnored implements go list's exclusion rules. go help list:
// 		Directory and file names that begin with "." or "_" are ignored
// 		by the go tool, as are directories named "testdata".
func checkIgnored(suffix string) bool {
	for _, component := range strings.Split(suffix, string(filepath.Separator)) {
		if len(component) == 0 {
			continue
		}
		if component[0] == '.' || component[0] == '_' || component == "testdata" {
			return true
		}
	}
	return false
}

func (v *View) Snapshot(ctx context.Context) (source.Snapshot, func()) {
	v.snapshotMu.Lock()
	defer v.snapshotMu.Unlock()
	return v.snapshot, v.snapshot.generation.Acquire(ctx)
}

func (s *snapshot) initialize(ctx context.Context, firstAttempt bool) {
	select {
	case <-ctx.Done():
		return
	case s.view.initializationSema <- struct{}{}:
	}

	defer func() {
		<-s.view.initializationSema
	}()

	if s.view.initializeOnce == nil {
		return
	}
	s.view.initializeOnce.Do(func() {
		defer func() {
			s.view.initializeOnce = nil
			if firstAttempt {
				close(s.view.initialized)
			}
		}()

		// If we have multiple modules, we need to load them by paths.
		var scopes []interface{}
		var modErrors *source.ErrorList
		addError := func(uri span.URI, err error) {
			if modErrors == nil {
				modErrors = &source.ErrorList{}
			}
			*modErrors = append(*modErrors, &source.Error{
				URI:      uri,
				Category: "compiler",
				Kind:     source.ListError,
				Message:  err.Error(),
			})
		}
		if len(s.modules) > 0 {
			for _, mod := range s.modules {
				fh, err := s.GetFile(ctx, mod.modURI)
				if err != nil {
					addError(mod.modURI, err)
					continue
				}
				parsed, err := s.ParseMod(ctx, fh)
				if err != nil {
					addError(mod.modURI, err)
					continue
				}
				path := parsed.File.Module.Mod.Path
				scopes = append(scopes, moduleLoadScope(path))
			}
		}
		if len(scopes) == 0 {
			scopes = append(scopes, viewLoadScope("LOAD_VIEW"))
		}
		err := s.load(ctx, append(scopes, packagePath("builtin"))...)
		if ctx.Err() != nil {
			return
		}
		if err != nil {
			event.Error(ctx, "initial workspace load failed", err)
			if modErrors != nil {
				s.view.initializedErr = errors.Errorf("errors loading modules: %v: %w", err, modErrors)
			} else {
				s.view.initializedErr = err
			}
		}
	})
}

// invalidateContent invalidates the content of a Go file,
// including any position and type information that depends on it.
// It returns true if we were already tracking the given file, false otherwise.
func (v *View) invalidateContent(ctx context.Context, uris map[span.URI]source.VersionedFileHandle, forceReloadMetadata bool) (source.Snapshot, func()) {
	// Detach the context so that content invalidation cannot be canceled.
	ctx = xcontext.Detach(ctx)

	// Cancel all still-running previous requests, since they would be
	// operating on stale data.
	v.cancelBackground()

	// Do not clone a snapshot until its view has finished initializing.
	v.snapshot.AwaitInitialized(ctx)

	// This should be the only time we hold the view's snapshot lock for any period of time.
	v.snapshotMu.Lock()
	defer v.snapshotMu.Unlock()

	oldSnapshot := v.snapshot
	var reinitialize reinitializeView
	v.snapshot, reinitialize = oldSnapshot.clone(ctx, uris, forceReloadMetadata)
	go oldSnapshot.generation.Destroy()

	if reinitialize == maybeReinit || reinitialize == definitelyReinit {
		v.reinitialize(reinitialize == definitelyReinit)
	}

	return v.snapshot, v.snapshot.generation.Acquire(ctx)
}

func (v *View) cancelBackground() {
	v.mu.Lock()
	defer v.mu.Unlock()
	if v.cancel == nil {
		// this can happen during shutdown
		return
	}
	v.cancel()
	v.backgroundCtx, v.cancel = context.WithCancel(v.baseCtx)
}

func (v *View) reinitialize(force bool) {
	v.initializationSema <- struct{}{}
	defer func() {
		<-v.initializationSema
	}()

	if !force && v.initializedErr == nil {
		return
	}
	var once sync.Once
	v.initializeOnce = &once
}

func (s *Session) getWorkspaceInformation(ctx context.Context, folder span.URI, options *source.Options) (*workspaceInformation, error) {
	if err := checkPathCase(folder.Filename()); err != nil {
		return nil, errors.Errorf("invalid workspace configuration: %w", err)
	}
	var err error
	goversion, err := s.goVersion(ctx, folder.Filename(), options.Env)
	if err != nil {
		return nil, err
	}

	go111module := os.Getenv("GO111MODULE")
	for _, kv := range options.Env {
		split := strings.SplitN(kv, "=", 2)
		if len(split) != 2 {
			continue
		}
		if split[0] == "GO111MODULE" {
			go111module = split[1]
		}
	}
	// If using 1.16, change the default back to auto. The primary effect of
	// GO111MODULE=on is to break GOPATH, which we aren't too interested in.
	if goversion >= 16 && go111module == "" {
		go111module = "auto"
	}

	// Make sure to get the `go env` before continuing with initialization.
	envVars, env, err := s.getGoEnv(ctx, folder.Filename(), append(options.Env, "GO111MODULE="+go111module))
	if err != nil {
		return nil, err
	}
	// The value of GOPACKAGESDRIVER is not returned through the go command.
	gopackagesdriver := os.Getenv("GOPACKAGESDRIVER")
	for _, s := range env {
		split := strings.SplitN(s, "=", 2)
		if split[0] == "GOPACKAGESDRIVER" {
			gopackagesdriver = split[1]
		}
	}
	// A user may also have a gopackagesdriver binary on their machine, which
	// works the same way as setting GOPACKAGESDRIVER.
	tool, _ := exec.LookPath("gopackagesdriver")
	hasGopackagesDriver := gopackagesdriver != "off" && (gopackagesdriver != "" || tool != "")

	var modURI span.URI
	if envVars.gomod != os.DevNull && envVars.gomod != "" {
		modURI = span.URIFromPath(envVars.gomod)
	}
	root := folder
	if options.ExpandWorkspaceToModule && modURI != "" {
		root = span.URIFromPath(filepath.Dir(modURI.Filename()))
	}
	return &workspaceInformation{
		hasGopackagesDriver:  hasGopackagesDriver,
		go111module:          go111module,
		goversion:            goversion,
		rootURI:              root,
		environmentVariables: envVars,
		goEnv:                env,
	}, nil
}

// OS-specific path case check, for case-insensitive filesystems.
var checkPathCase = defaultCheckPathCase

func defaultCheckPathCase(path string) error {
	return nil
}

func validBuildConfiguration(folder span.URI, ws *workspaceInformation, modules map[span.URI]*moduleRoot) bool {
	// Since we only really understand the `go` command, if the user has a
	// different GOPACKAGESDRIVER, assume that their configuration is valid.
	if ws.hasGopackagesDriver {
		return true
	}
	// Check if the user is working within a module or if we have found
	// multiple modules in the workspace.
	if len(modules) > 0 {
		return true
	}
	// The user may have a multiple directories in their GOPATH.
	// Check if the workspace is within any of them.
	for _, gp := range filepath.SplitList(ws.gopath) {
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

// getGoEnv gets the view's various GO* values.
func (s *Session) getGoEnv(ctx context.Context, folder string, configEnv []string) (environmentVariables, map[string]string, error) {
	envVars := environmentVariables{}
	vars := map[string]*string{
		"GOCACHE":    &envVars.gocache,
		"GOPATH":     &envVars.gopath,
		"GOPRIVATE":  &envVars.goprivate,
		"GOMODCACHE": &envVars.gomodcache,
		"GOMOD":      &envVars.gomod,
	}
	// We can save ~200 ms by requesting only the variables we care about.
	args := append([]string{"-json"}, imports.RequiredGoEnvVars...)
	for k := range vars {
		args = append(args, k)
	}

	inv := gocommand.Invocation{
		Verb:       "env",
		Args:       args,
		Env:        configEnv,
		WorkingDir: folder,
	}
	// Don't go through runGoCommand, as we don't need a temporary -modfile to
	// run `go env`.
	stdout, err := s.gocmdRunner.Run(ctx, inv)
	if err != nil {
		return environmentVariables{}, nil, err
	}
	env := make(map[string]string)
	if err := json.Unmarshal(stdout.Bytes(), &env); err != nil {
		return environmentVariables{}, nil, err
	}

	for key, ptr := range vars {
		*ptr = env[key]
	}

	// Old versions of Go don't have GOMODCACHE, so emulate it.
	if envVars.gomodcache == "" && envVars.gopath != "" {
		envVars.gomodcache = filepath.Join(filepath.SplitList(envVars.gopath)[0], "pkg/mod")
	}
	return envVars, env, err
}

func (v *View) IsGoPrivatePath(target string) bool {
	return globsMatchPath(v.goprivate, target)
}

// Copied from
// https://cs.opensource.google/go/go/+/master:src/cmd/go/internal/str/path.go;l=58;drc=2910c5b4a01a573ebc97744890a07c1a3122c67a
func globsMatchPath(globs, target string) bool {
	for globs != "" {
		// Extract next non-empty glob in comma-separated list.
		var glob string
		if i := strings.Index(globs, ","); i >= 0 {
			glob, globs = globs[:i], globs[i+1:]
		} else {
			glob, globs = globs, ""
		}
		if glob == "" {
			continue
		}

		// A glob with N+1 path elements (N slashes) needs to be matched
		// against the first N+1 path elements of target,
		// which end just before the N+1'th slash.
		n := strings.Count(glob, "/")
		prefix := target
		// Walk target, counting slashes, truncating at the N+1'th slash.
		for i := 0; i < len(target); i++ {
			if target[i] == '/' {
				if n == 0 {
					prefix = target[:i]
					break
				}
				n--
			}
		}
		if n > 0 {
			// Not enough prefix elements.
			continue
		}
		matched, _ := path.Match(glob, prefix)
		if matched {
			return true
		}
	}
	return false
}

var modFlagRegexp = regexp.MustCompile(`-mod[ =](\w+)`)

// TODO(rstambler): Consolidate modURI and modContent back into a FileHandle
// after we have a version of the workspace go.mod file on disk. Getting a
// FileHandle from the cache for temporary files is problematic, since we
// cannot delete it.
func (s *snapshot) needsModEqualsMod(ctx context.Context, modURI span.URI, modContent []byte) (bool, error) {
	if s.view.goversion < 16 || s.workspaceMode()&moduleMode == 0 {
		return false, nil
	}

	matches := modFlagRegexp.FindStringSubmatch(s.view.goEnv["GOFLAGS"])
	var modFlag string
	if len(matches) != 0 {
		modFlag = matches[1]
	}
	if modFlag != "" {
		// Don't override an explicit '-mod=vendor' argument.
		// We do want to override '-mod=readonly': it would break various module code lenses,
		// and on 1.16 we know -modfile is available, so we won't mess with go.mod anyway.
		return modFlag == "vendor", nil
	}

	modFile, err := modfile.Parse(modURI.Filename(), modContent, nil)
	if err != nil {
		return false, err
	}
	if fi, err := os.Stat(filepath.Join(s.view.rootURI.Filename(), "vendor")); err != nil || !fi.IsDir() {
		return true, nil
	}
	vendorEnabled := modFile.Go != nil && modFile.Go.Version != "" && semver.Compare("v"+modFile.Go.Version, "v1.14") >= 0
	return !vendorEnabled, nil
}

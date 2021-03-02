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
	"path"
	"path/filepath"
	"reflect"
	"regexp"
	"sort"
	"strings"
	"sync"

	"golang.org/x/mod/modfile"
	"golang.org/x/mod/semver"
	exec "golang.org/x/sys/execabs"
	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/gocommand"
	"golang.org/x/tools/internal/imports"
	"golang.org/x/tools/internal/lsp/protocol"
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

	// cancel is called when all action being performed by the current view
	// should be stopped.
	cancel context.CancelFunc

	// name is the user visible name of this view.
	name string

	// folder is the folder with which this view was constructed.
	folder span.URI

	importsState *importsState

	// moduleUpgrades tracks known upgrades for module paths.
	moduleUpgrades map[string]string

	// keep track of files by uri and by basename, a single file may be mapped
	// to multiple uris, and the same basename may map to multiple files
	filesByURI  map[span.URI]*fileBase
	filesByBase map[string][]*fileBase

	// initCancelFirstAttempt can be used to terminate the view's first
	// attempt at initialization.
	initCancelFirstAttempt context.CancelFunc

	snapshotMu sync.Mutex
	snapshot   *snapshot

	// initialWorkspaceLoad is closed when the first workspace initialization has
	// completed. If we failed to load, we only retry if the go.mod file changes,
	// to avoid too many go/packages calls.
	initialWorkspaceLoad chan struct{}

	// initializationSema is used limit concurrent initialization of snapshots in
	// the view. We use a channel instead of a mutex to avoid blocking when a
	// context is canceled.
	initializationSema chan struct{}

	// rootURI is the rootURI directory of this view. If we are in GOPATH mode, this
	// is just the folder. If we are in module mode, this is the module rootURI.
	rootURI span.URI

	// workspaceInformation tracks various details about this view's
	// environment variables, go version, and use of modules.
	workspaceInformation

	// tempWorkspace is a temporary directory dedicated to holding the latest
	// version of the workspace go.mod file. (TODO: also go.sum file)
	tempWorkspace span.URI
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

	// userGo111Module is the user's value of GO111MODULE.
	userGo111Module go111module

	// The value of GO111MODULE we want to run with.
	effectiveGo111Module string

	// goEnv is the `go env` output collected when a view is created.
	// It includes the values of the environment variables above.
	goEnv map[string]string
}

type go111module int

const (
	off = go111module(iota)
	auto
	on
)

type environmentVariables struct {
	gocache, gopath, goroot, goprivate, gomodcache, go111module string
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
func tempModFile(modFh source.FileHandle, gosum []byte) (tmpURI span.URI, cleanup func(), err error) {
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
	if gosum != nil {
		if err := ioutil.WriteFile(tmpSumName, gosum, 0655); err != nil {
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

func (v *View) TempWorkspace() span.URI {
	return v.tempWorkspace
}

func (v *View) Options() *source.Options {
	v.optionsMu.Lock()
	defer v.optionsMu.Unlock()
	return v.options
}

func minorOptionsChange(a, b *source.Options) bool {
	// Check if any of the settings that modify our understanding of files have been changed
	if !reflect.DeepEqual(a.Env, b.Env) {
		return false
	}
	if !reflect.DeepEqual(a.DirectoryFilters, b.DirectoryFilters) {
		return false
	}
	if a.MemoryMode != b.MemoryMode {
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
	env := s.view.options.EnvSlice()
	buildFlags := append([]string{}, s.view.options.BuildFlags...)
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

func (s *snapshot) locateTemplateFiles(ctx context.Context) {
	if !s.view.Options().ExperimentalTemplateSupport {
		return
	}
	dir := s.workspace.root.Filename()
	searched := 0
	// Change to WalkDir when we move up to 1.16
	err := filepath.Walk(dir, func(path string, fi os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if strings.HasSuffix(filepath.Ext(path), "tmpl") && !pathExcludedByFilter(path, s.view.options) &&
			!fi.IsDir() {
			k := span.URIFromPath(path)
			fh, err := s.GetVersionedFile(ctx, k)
			if err != nil {
				return nil
			}
			s.files[k] = fh
		}
		searched++
		if fileLimit > 0 && searched > fileLimit {
			return errExhausted
		}
		return nil
	})
	if err != nil {
		event.Error(ctx, "searching for template files failed", err)
	}
}

func (v *View) contains(uri span.URI) bool {
	inRoot := source.InDir(v.rootURI.Filename(), uri.Filename())
	inFolder := source.InDir(v.folder.Filename(), uri.Filename())
	if !inRoot && !inFolder {
		return false
	}
	// Filters are applied relative to the workspace folder.
	if inFolder {
		return !pathExcludedByFilter(strings.TrimPrefix(uri.Filename(), v.folder.Filename()), v.Options())
	}
	return true
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
	if v.knownFile(c.URI) {
		return true
	}
	// The gopls.mod may not be "known" because we first access it through the
	// session. As a result, treat changes to the view's gopls.mod file as
	// always relevant, even if they are only on-disk changes.
	// TODO(rstambler): Make sure the gopls.mod is always known to the view.
	if c.URI == goplsModURI(v.rootURI) {
		return true
	}
	// If the file is not known to the view, and the change is only on-disk,
	// we should not invalidate the snapshot. This is necessary because Emacs
	// sends didChangeWatchedFiles events for temp files.
	if c.OnDisk && (c.Action == source.Change || c.Action == source.Delete) {
		return false
	}
	return v.contains(c.URI)
}

func (v *View) knownFile(uri span.URI) bool {
	v.mu.Lock()
	defer v.mu.Unlock()

	f, err := v.findFile(uri)
	return f != nil && err == nil
}

// getFile returns a file for the given URI.
func (v *View) getFile(uri span.URI) *fileBase {
	v.mu.Lock()
	defer v.mu.Unlock()

	f, _ := v.findFile(uri)
	if f != nil {
		return f
	}
	f = &fileBase{
		view:  v,
		fname: uri.Filename(),
	}
	v.mapFile(uri, f)
	return f
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

// TODO(rFindley): probably some of this should also be one in View.Shutdown
// above?
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
	v.importsState.destroy()
}

func (v *View) Session() *Session {
	return v.session
}

func (s *snapshot) IgnoredFile(uri span.URI) bool {
	filename := uri.Filename()
	var prefixes []string
	if len(s.workspace.getActiveModFiles()) == 0 {
		for _, entry := range filepath.SplitList(s.view.gopath) {
			prefixes = append(prefixes, filepath.Join(entry, "src"))
		}
	} else {
		prefixes = append(prefixes, s.view.gomodcache)
		for m := range s.workspace.getActiveModFiles() {
			prefixes = append(prefixes, dirURI(m).Filename())
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
	return v.getSnapshot(ctx)
}

func (v *View) getSnapshot(ctx context.Context) (*snapshot, func()) {
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

	if s.initializeOnce == nil {
		return
	}
	s.initializeOnce.Do(func() {
		s.loadWorkspace(ctx, firstAttempt)
	})
}

func (s *snapshot) loadWorkspace(ctx context.Context, firstAttempt bool) {
	defer func() {
		s.initializeOnce = nil
		if firstAttempt {
			close(s.view.initialWorkspaceLoad)
		}
	}()

	// If we have multiple modules, we need to load them by paths.
	var scopes []interface{}
	var modDiagnostics []*source.Diagnostic
	addError := func(uri span.URI, err error) {
		modDiagnostics = append(modDiagnostics, &source.Diagnostic{
			URI:      uri,
			Severity: protocol.SeverityError,
			Source:   source.ListError,
			Message:  err.Error(),
		})
	}
	s.locateTemplateFiles(ctx)
	if len(s.workspace.getActiveModFiles()) > 0 {
		for modURI := range s.workspace.getActiveModFiles() {
			fh, err := s.GetFile(ctx, modURI)
			if err != nil {
				addError(modURI, err)
				continue
			}
			parsed, err := s.ParseMod(ctx, fh)
			if err != nil {
				addError(modURI, err)
				continue
			}
			if parsed.File == nil || parsed.File.Module == nil {
				addError(modURI, fmt.Errorf("no module path for %s", modURI))
				continue
			}
			path := parsed.File.Module.Mod.Path
			scopes = append(scopes, moduleLoadScope(path))
		}
	} else {
		scopes = append(scopes, viewLoadScope("LOAD_VIEW"))
	}
	var err error
	if len(scopes) > 0 {
		err = s.load(ctx, firstAttempt, append(scopes, packagePath("builtin"))...)
	}
	if ctx.Err() != nil {
		return
	}

	var criticalErr *source.CriticalError
	if err != nil {
		event.Error(ctx, "initial workspace load failed", err)
		extractedDiags, _ := s.extractGoCommandErrors(ctx, err.Error())
		criticalErr = &source.CriticalError{
			MainError: err,
			DiagList:  append(modDiagnostics, extractedDiags...),
		}
	} else if len(modDiagnostics) == 1 {
		criticalErr = &source.CriticalError{
			MainError: fmt.Errorf(modDiagnostics[0].Message),
			DiagList:  modDiagnostics,
		}
	} else if len(modDiagnostics) > 1 {
		criticalErr = &source.CriticalError{
			MainError: fmt.Errorf("error loading module names"),
			DiagList:  modDiagnostics,
		}
	}

	// Lock the snapshot when setting the initialized error.
	s.mu.Lock()
	defer s.mu.Unlock()
	s.initializedErr = criticalErr
}

// invalidateContent invalidates the content of a Go file,
// including any position and type information that depends on it.
func (v *View) invalidateContent(ctx context.Context, changes map[span.URI]*fileChange, forceReloadMetadata bool) (*snapshot, func()) {
	// Detach the context so that content invalidation cannot be canceled.
	ctx = xcontext.Detach(ctx)

	// Cancel all still-running previous requests, since they would be
	// operating on stale data.
	v.snapshot.cancel()

	// Do not clone a snapshot until its view has finished initializing.
	v.snapshot.AwaitInitialized(ctx)

	// This should be the only time we hold the view's snapshot lock for any period of time.
	v.snapshotMu.Lock()
	defer v.snapshotMu.Unlock()

	oldSnapshot := v.snapshot

	var workspaceChanged bool
	v.snapshot, workspaceChanged = oldSnapshot.clone(ctx, v.baseCtx, changes, forceReloadMetadata)
	if workspaceChanged && v.tempWorkspace != "" {
		snap := v.snapshot
		release := snap.generation.Acquire(ctx)
		go func() {
			defer release()
			wsdir, err := snap.getWorkspaceDir(ctx)
			if err != nil {
				event.Error(ctx, "getting workspace dir", err)
			}
			if err := copyWorkspace(v.tempWorkspace, wsdir); err != nil {
				event.Error(ctx, "copying workspace dir", err)
			}
		}()
	}
	go oldSnapshot.generation.Destroy()

	return v.snapshot, v.snapshot.generation.Acquire(ctx)
}

func copyWorkspace(dst span.URI, src span.URI) error {
	for _, name := range []string{"go.mod", "go.sum"} {
		srcname := filepath.Join(src.Filename(), name)
		srcf, err := os.Open(srcname)
		if err != nil {
			return errors.Errorf("opening snapshot %s: %w", name, err)
		}
		defer srcf.Close()
		dstname := filepath.Join(dst.Filename(), name)
		dstf, err := os.Create(dstname)
		if err != nil {
			return errors.Errorf("truncating view %s: %w", name, err)
		}
		defer dstf.Close()
		if _, err := io.Copy(dstf, srcf); err != nil {
			return errors.Errorf("copying %s: %w", name, err)
		}
	}
	return nil
}

func (s *Session) getWorkspaceInformation(ctx context.Context, folder span.URI, options *source.Options) (*workspaceInformation, error) {
	if err := checkPathCase(folder.Filename()); err != nil {
		return nil, errors.Errorf("invalid workspace configuration: %w", err)
	}
	var err error
	inv := gocommand.Invocation{
		WorkingDir: folder.Filename(),
		Env:        options.EnvSlice(),
	}
	goversion, err := gocommand.GoVersion(ctx, inv, s.gocmdRunner)
	if err != nil {
		return nil, err
	}

	go111module := os.Getenv("GO111MODULE")
	if v, ok := options.Env["GO111MODULE"]; ok {
		go111module = v
	}
	// Make sure to get the `go env` before continuing with initialization.
	envVars, env, err := s.getGoEnv(ctx, folder.Filename(), goversion, go111module, options.EnvSlice())
	if err != nil {
		return nil, err
	}
	// If using 1.16, change the default back to auto. The primary effect of
	// GO111MODULE=on is to break GOPATH, which we aren't too interested in.
	if goversion >= 16 && go111module == "" {
		go111module = "auto"
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

	return &workspaceInformation{
		hasGopackagesDriver:  hasGopackagesDriver,
		effectiveGo111Module: go111module,
		userGo111Module:      go111moduleForVersion(go111module, goversion),
		goversion:            goversion,
		environmentVariables: envVars,
		goEnv:                env,
	}, nil
}

func go111moduleForVersion(go111module string, goversion int) go111module {
	// Off by default until Go 1.12.
	if go111module == "off" || (goversion < 12 && go111module == "") {
		return off
	}
	// On by default as of Go 1.16.
	if go111module == "on" || (goversion >= 16 && go111module == "") {
		return on
	}
	return auto
}

// findWorkspaceRoot searches for the best workspace root according to the
// following heuristics:
//   - First, look for a parent directory containing a gopls.mod file
//     (experimental only).
//   - Then, a parent directory containing a go.mod file.
//   - Then, a child directory containing a go.mod file, if there is exactly
//     one (non-experimental only).
// Otherwise, it returns folder.
// TODO (rFindley): move this to workspace.go
// TODO (rFindley): simplify this once workspace modules are enabled by default.
func findWorkspaceRoot(ctx context.Context, folder span.URI, fs source.FileSource, excludePath func(string) bool, experimental bool) (span.URI, error) {
	patterns := []string{"go.mod"}
	if experimental {
		patterns = []string{"gopls.mod", "go.mod"}
	}
	for _, basename := range patterns {
		dir, err := findRootPattern(ctx, folder, basename, fs)
		if err != nil {
			return "", errors.Errorf("finding %s: %w", basename, err)
		}
		if dir != "" {
			return dir, nil
		}
	}

	// The experimental workspace can handle nested modules at this point...
	if experimental {
		return folder, nil
	}

	// ...else we should check if there's exactly one nested module.
	all, err := findModules(folder, excludePath, 2)
	if err == errExhausted {
		// Fall-back behavior: if we don't find any modules after searching 10000
		// files, assume there are none.
		event.Log(ctx, fmt.Sprintf("stopped searching for modules after %d files", fileLimit))
		return folder, nil
	}
	if err != nil {
		return "", err
	}
	if len(all) == 1 {
		// range to access first element.
		for uri := range all {
			return dirURI(uri), nil
		}
	}
	return folder, nil
}

func findRootPattern(ctx context.Context, folder span.URI, basename string, fs source.FileSource) (span.URI, error) {
	dir := folder.Filename()
	for dir != "" {
		target := filepath.Join(dir, basename)
		exists, err := fileExists(ctx, span.URIFromPath(target), fs)
		if err != nil {
			return "", err
		}
		if exists {
			return span.URIFromPath(dir), nil
		}
		next, _ := filepath.Split(dir)
		if next == dir {
			break
		}
		dir = next
	}
	return "", nil
}

// OS-specific path case check, for case-insensitive filesystems.
var checkPathCase = defaultCheckPathCase

func defaultCheckPathCase(path string) error {
	return nil
}

func validBuildConfiguration(folder span.URI, ws *workspaceInformation, modFiles map[span.URI]struct{}) bool {
	// Since we only really understand the `go` command, if the user has a
	// different GOPACKAGESDRIVER, assume that their configuration is valid.
	if ws.hasGopackagesDriver {
		return true
	}
	// Check if the user is working within a module or if we have found
	// multiple modules in the workspace.
	if len(modFiles) > 0 {
		return true
	}
	// The user may have a multiple directories in their GOPATH.
	// Check if the workspace is within any of them.
	for _, gp := range filepath.SplitList(ws.gopath) {
		if source.InDir(filepath.Join(gp, "src"), folder.Filename()) {
			return true
		}
	}
	return false
}

// getGoEnv gets the view's various GO* values.
func (s *Session) getGoEnv(ctx context.Context, folder string, goversion int, go111module string, configEnv []string) (environmentVariables, map[string]string, error) {
	envVars := environmentVariables{}
	vars := map[string]*string{
		"GOCACHE":     &envVars.gocache,
		"GOPATH":      &envVars.gopath,
		"GOROOT":      &envVars.goroot,
		"GOPRIVATE":   &envVars.goprivate,
		"GOMODCACHE":  &envVars.gomodcache,
		"GO111MODULE": &envVars.go111module,
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
	// GO111MODULE does not appear in `go env` output until Go 1.13.
	if goversion < 13 {
		envVars.go111module = go111module
	}
	return envVars, env, err
}

func (v *View) IsGoPrivatePath(target string) bool {
	return globsMatchPath(v.goprivate, target)
}

func (v *View) ModuleUpgrades() map[string]string {
	v.mu.Lock()
	defer v.mu.Unlock()

	upgrades := map[string]string{}
	for mod, ver := range v.moduleUpgrades {
		upgrades[mod] = ver
	}
	return upgrades
}

func (v *View) RegisterModuleUpgrades(upgrades map[string]string) {
	v.mu.Lock()
	defer v.mu.Unlock()

	for mod, ver := range upgrades {
		v.moduleUpgrades[mod] = ver
	}
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
func (s *snapshot) vendorEnabled(ctx context.Context, modURI span.URI, modContent []byte) (bool, error) {
	if s.workspaceMode()&moduleMode == 0 {
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
		return false, nil
	}
	vendorEnabled := modFile.Go != nil && modFile.Go.Version != "" && semver.Compare("v"+modFile.Go.Version, "v1.14") >= 0
	return vendorEnabled, nil
}

func (v *View) allFilesExcluded(pkg *packages.Package) bool {
	opts := v.Options()
	folder := filepath.ToSlash(v.folder.Filename())
	for _, f := range pkg.GoFiles {
		f = filepath.ToSlash(f)
		if !strings.HasPrefix(f, folder) {
			return false
		}
		if !pathExcludedByFilter(strings.TrimPrefix(f, folder), opts) {
			return false
		}
	}
	return true
}

func pathExcludedByFilterFunc(opts *source.Options) func(string) bool {
	return func(path string) bool {
		return pathExcludedByFilter(path, opts)
	}
}

func pathExcludedByFilter(path string, opts *source.Options) bool {
	path = strings.TrimPrefix(filepath.ToSlash(path), "/")

	excluded := false
	for _, filter := range opts.DirectoryFilters {
		op, prefix := filter[0], filter[1:]
		// Non-empty prefixes have to be precise directory matches.
		if prefix != "" {
			prefix = prefix + "/"
			path = path + "/"
		}
		if !strings.HasPrefix(path, prefix) {
			continue
		}
		excluded = op == '-'
	}
	return excluded
}

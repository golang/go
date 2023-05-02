// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package cache implements the caching layer for gopls.
package cache

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"os"
	"path"
	"path/filepath"
	"reflect"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"

	"golang.org/x/mod/modfile"
	"golang.org/x/mod/semver"
	exec "golang.org/x/sys/execabs"
	"golang.org/x/tools/gopls/internal/govulncheck"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/gopls/internal/span"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/gocommand"
	"golang.org/x/tools/internal/imports"
	"golang.org/x/tools/internal/xcontext"
)

type View struct {
	id string

	gocmdRunner *gocommand.Runner // limits go command concurrency

	// baseCtx is the context handed to NewView. This is the parent of all
	// background contexts created for this view.
	baseCtx context.Context

	// name is the user-specified name of this view.
	name string

	optionsMu sync.Mutex
	options   *source.Options

	// Workspace information. The fields below are immutable, and together with
	// options define the build list. Any change to these fields results in a new
	// View.
	folder               span.URI // user-specified workspace folder
	workspaceInformation          // Go environment information

	importsState *importsState

	// moduleUpgrades tracks known upgrades for module paths in each modfile.
	// Each modfile has a map of module name to upgrade version.
	moduleUpgradesMu sync.Mutex
	moduleUpgrades   map[span.URI]map[string]string

	// vulns maps each go.mod file's URI to its known vulnerabilities.
	vulnsMu sync.Mutex
	vulns   map[span.URI]*govulncheck.Result

	// fs is the file source used to populate this view.
	fs *overlayFS

	// seenFiles tracks files that the view has accessed.
	// TODO(golang/go#57558): this notion is fundamentally problematic, and
	// should be removed.
	knownFilesMu sync.Mutex
	knownFiles   map[span.URI]bool

	// initCancelFirstAttempt can be used to terminate the view's first
	// attempt at initialization.
	initCancelFirstAttempt context.CancelFunc

	// Track the latest snapshot via the snapshot field, guarded by snapshotMu.
	//
	// Invariant: whenever the snapshot field is overwritten, destroy(snapshot)
	// is called on the previous (overwritten) snapshot while snapshotMu is held,
	// incrementing snapshotWG. During shutdown the final snapshot is
	// overwritten with nil and destroyed, guaranteeing that all observed
	// snapshots have been destroyed via the destroy method, and snapshotWG may
	// be waited upon to let these destroy operations complete.
	snapshotMu      sync.Mutex
	snapshot        *snapshot      // latest snapshot; nil after shutdown has been called
	releaseSnapshot func()         // called when snapshot is no longer needed
	snapshotWG      sync.WaitGroup // refcount for pending destroy operations

	// initialWorkspaceLoad is closed when the first workspace initialization has
	// completed. If we failed to load, we only retry if the go.mod file changes,
	// to avoid too many go/packages calls.
	initialWorkspaceLoad chan struct{}

	// initializationSema is used limit concurrent initialization of snapshots in
	// the view. We use a channel instead of a mutex to avoid blocking when a
	// context is canceled.
	//
	// This field (along with snapshot.initialized) guards against duplicate
	// initialization of snapshots. Do not change it without adjusting snapshot
	// accordingly.
	initializationSema chan struct{}
}

// workspaceInformation holds the defining features of the View workspace.
//
// This type is compared to see if the View needs to be reconstructed.
type workspaceInformation struct {
	// `go env` variables that need to be tracked by gopls.
	goEnv

	// gomod holds the relevant go.mod file for this workspace.
	gomod span.URI

	// The Go version in use: X in Go 1.X.
	goversion int

	// The complete output of the go version command.
	// (Call gocommand.ParseGoVersionOutput to extract a version
	// substring such as go1.19.1 or go1.20-rc.1, go1.21-abcdef01.)
	goversionOutput string

	// hasGopackagesDriver is true if the user has a value set for the
	// GOPACKAGESDRIVER environment variable or a gopackagesdriver binary on
	// their machine.
	hasGopackagesDriver bool

	// inGOPATH reports whether the workspace directory is contained in a GOPATH
	// directory.
	inGOPATH bool
}

// effectiveGO111MODULE reports the value of GO111MODULE effective in the go
// command at this go version, assuming at least Go 1.16.
func (w workspaceInformation) effectiveGO111MODULE() go111module {
	switch w.GO111MODULE() {
	case "off":
		return off
	case "on", "":
		return on
	default:
		return auto
	}
}

// A ViewType describes how we load package information for a view.
//
// This is used for constructing the go/packages.Load query, and for
// interpreting missing packages, imports, or errors.
//
// Each view has a ViewType which is derived from its immutable workspace
// information -- any environment change that would affect the view type
// results in a new view.
type ViewType int

const (
	// GoPackagesDriverView is a view with a non-empty GOPACKAGESDRIVER
	// environment variable.
	GoPackagesDriverView ViewType = iota

	// GOPATHView is a view in GOPATH mode.
	//
	// I.e. in GOPATH, with GO111MODULE=off, or GO111MODULE=auto with no
	// go.mod file.
	GOPATHView

	// GoModuleView is a view in module mode with a single Go module.
	GoModuleView

	// GoWorkView is a view in module mode with a go.work file.
	GoWorkView

	// An AdHocView is a collection of files in a given directory, not in GOPATH
	// or a module.
	AdHocView
)

// ViewType derives the type of the view from its workspace information.
//
// TODO(rfindley): this logic is overlapping and slightly inconsistent with
// validBuildConfiguration. As part of zero-config-gopls (golang/go#57979), fix
// this inconsistency and consolidate on the ViewType abstraction.
func (w workspaceInformation) ViewType() ViewType {
	if w.hasGopackagesDriver {
		return GoPackagesDriverView
	}
	go111module := w.effectiveGO111MODULE()
	if w.gowork != "" && go111module != off {
		return GoWorkView
	}
	if w.gomod != "" && go111module != off {
		return GoModuleView
	}
	if w.inGOPATH && go111module != on {
		return GOPATHView
	}
	return AdHocView
}

// moduleMode reports whether the current snapshot uses Go modules.
//
// From https://go.dev/ref/mod, module mode is active if either of the
// following hold:
//   - GO111MODULE=on
//   - GO111MODULE=auto and we are inside a module or have a GOWORK value.
//
// Additionally, this method returns false if GOPACKAGESDRIVER is set.
//
// TODO(rfindley): use this more widely.
func (w workspaceInformation) moduleMode() bool {
	switch w.ViewType() {
	case GoModuleView, GoWorkView:
		return true
	default:
		return false
	}
}

// GOWORK returns the effective GOWORK value for this workspace, if
// any, in URI form.
//
// The second result reports whether the effective GOWORK value is "" because
// GOWORK=off.
func (w workspaceInformation) GOWORK() (span.URI, bool) {
	if w.gowork == "off" || w.gowork == "" {
		return "", w.gowork == "off"
	}
	return span.URIFromPath(w.gowork), false
}

// GO111MODULE returns the value of GO111MODULE to use for running the go
// command. It differs from the user's environment in order to allow for the
// more forgiving default value "auto" when using recent go versions.
//
// TODO(rfindley): it is probably not worthwhile diverging from the go command
// here. The extra forgiveness may be nice, but breaks the invariant that
// running the go command from the command line produces the same build list.
//
// Put differently: we shouldn't go out of our way to make GOPATH work, when
// the go command does not.
func (w workspaceInformation) GO111MODULE() string {
	if w.go111module == "" {
		return "auto"
	}
	return w.go111module
}

type go111module int

const (
	off = go111module(iota)
	auto
	on
)

// goEnv holds important environment variables that gopls cares about.
type goEnv struct {
	gocache, gopath, goroot, goprivate, gomodcache, gowork, goflags string

	// go111module holds the value of GO111MODULE as reported by go env.
	//
	// Don't use this value directly, because we choose to use a different
	// default (auto) on Go 1.16 and later, to avoid spurious errors. Use
	// the effectiveGO111MODULE method instead.
	go111module string
}

// loadGoEnv loads `go env` values into the receiver, using the provided user
// environment and go command runner.
func (env *goEnv) load(ctx context.Context, folder string, configEnv []string, runner *gocommand.Runner) error {
	vars := env.vars()

	// We can save ~200 ms by requesting only the variables we care about.
	args := []string{"-json"}
	for k := range vars {
		args = append(args, k)
	}

	inv := gocommand.Invocation{
		Verb:       "env",
		Args:       args,
		Env:        configEnv,
		WorkingDir: folder,
	}
	stdout, err := runner.Run(ctx, inv)
	if err != nil {
		return err
	}
	envMap := make(map[string]string)
	if err := json.Unmarshal(stdout.Bytes(), &envMap); err != nil {
		return fmt.Errorf("internal error unmarshaling JSON from 'go env': %w", err)
	}
	for key, ptr := range vars {
		*ptr = envMap[key]
	}

	return nil
}

func (env goEnv) String() string {
	var vars []string
	for govar, ptr := range env.vars() {
		vars = append(vars, fmt.Sprintf("%s=%s", govar, *ptr))
	}
	sort.Strings(vars)
	return "[" + strings.Join(vars, ", ") + "]"
}

// vars returns a map from Go environment variable to field value containing it.
func (env *goEnv) vars() map[string]*string {
	return map[string]*string{
		"GOCACHE":     &env.gocache,
		"GOPATH":      &env.gopath,
		"GOROOT":      &env.goroot,
		"GOPRIVATE":   &env.goprivate,
		"GOMODCACHE":  &env.gomodcache,
		"GO111MODULE": &env.go111module,
		"GOWORK":      &env.gowork,
		"GOFLAGS":     &env.goflags,
	}
}

// workspaceMode holds various flags defining how the gopls workspace should
// behave. They may be derived from the environment, user configuration, or
// depend on the Go version.
//
// TODO(rfindley): remove workspace mode, in favor of explicit checks.
type workspaceMode int

const (
	moduleMode workspaceMode = 1 << iota

	// tempModfile indicates whether or not the -modfile flag should be used.
	tempModfile
)

func (v *View) ID() string { return v.id }

// tempModFile creates a temporary go.mod file based on the contents
// of the given go.mod file. On success, it is the caller's
// responsibility to call the cleanup function when the file is no
// longer needed.
func tempModFile(modFh source.FileHandle, gosum []byte) (tmpURI span.URI, cleanup func(), err error) {
	filenameHash := source.Hashf("%s", modFh.URI().Filename())
	tmpMod, err := ioutil.TempFile("", fmt.Sprintf("go.%s.*.mod", filenameHash))
	if err != nil {
		return "", nil, err
	}
	defer tmpMod.Close()

	tmpURI = span.URIFromPath(tmpMod.Name())
	tmpSumName := sumFilename(tmpURI)

	content, err := modFh.Content()
	if err != nil {
		return "", nil, err
	}

	if _, err := tmpMod.Write(content); err != nil {
		return "", nil, err
	}

	// We use a distinct name here to avoid subtlety around the fact
	// that both 'return' and 'defer' update the "cleanup" variable.
	doCleanup := func() {
		_ = os.Remove(tmpSumName)
		_ = os.Remove(tmpURI.Filename())
	}

	// Be careful to clean up if we return an error from this function.
	defer func() {
		if err != nil {
			doCleanup()
			cleanup = nil
		}
	}()

	// Create an analogous go.sum, if one exists.
	if gosum != nil {
		if err := ioutil.WriteFile(tmpSumName, gosum, 0655); err != nil {
			return "", nil, err
		}
	}

	return tmpURI, doCleanup, nil
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

func (v *View) FileKind(fh source.FileHandle) source.FileKind {
	// The kind of an unsaved buffer comes from the
	// TextDocumentItem.LanguageID field in the didChange event,
	// not from the file name. They may differ.
	if o, ok := fh.(*Overlay); ok {
		if o.kind != source.UnknownKind {
			return o.kind
		}
	}

	fext := filepath.Ext(fh.URI().Filename())
	switch fext {
	case ".go":
		return source.Go
	case ".mod":
		return source.Mod
	case ".sum":
		return source.Sum
	case ".work":
		return source.Work
	}
	exts := v.Options().TemplateExtensions
	for _, ext := range exts {
		if fext == ext || fext == "."+ext {
			return source.Tmpl
		}
	}
	// and now what? This should never happen, but it does for cgo before go1.15
	return source.Go
}

func minorOptionsChange(a, b *source.Options) bool {
	// Check if any of the settings that modify our understanding of files have
	// been changed.
	if !reflect.DeepEqual(a.Env, b.Env) {
		return false
	}
	if !reflect.DeepEqual(a.DirectoryFilters, b.DirectoryFilters) {
		return false
	}
	if !reflect.DeepEqual(a.StandaloneTags, b.StandaloneTags) {
		return false
	}
	if a.ExpandWorkspaceToModule != b.ExpandWorkspaceToModule {
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

// SetViewOptions sets the options of the given view to new values. Calling
// this may cause the view to be invalidated and a replacement view added to
// the session. If so the new view will be returned, otherwise the original one
// will be returned.
func (s *Session) SetViewOptions(ctx context.Context, v *View, options *source.Options) (*View, error) {
	// no need to rebuild the view if the options were not materially changed
	v.optionsMu.Lock()
	if minorOptionsChange(v.options, options) {
		v.options = options
		v.optionsMu.Unlock()
		return v, nil
	}
	v.optionsMu.Unlock()
	newView, err := s.updateView(ctx, v, options)
	return newView, err
}

// viewEnv returns a string describing the environment of a newly created view.
//
// It must not be called concurrently with any other view methods.
func viewEnv(v *View) string {
	env := v.options.EnvSlice()
	buildFlags := append([]string{}, v.options.BuildFlags...)

	var buf bytes.Buffer
	fmt.Fprintf(&buf, `go info for %v
(go dir %s)
(go version %s)
(valid build configuration = %v)
(build flags: %v)
(selected go env: %v)
`,
		v.folder.Filename(),
		v.workingDir().Filename(),
		strings.TrimRight(v.workspaceInformation.goversionOutput, "\n"),
		v.snapshot.validBuildConfiguration(),
		buildFlags,
		v.goEnv,
	)

	for _, v := range env {
		s := strings.SplitN(v, "=", 2)
		if len(s) != 2 {
			continue
		}
	}

	return buf.String()
}

func (s *snapshot) RunProcessEnvFunc(ctx context.Context, fn func(context.Context, *imports.Options) error) error {
	return s.view.importsState.runProcessEnvFunc(ctx, s, fn)
}

// separated out from its sole use in locateTemplateFiles for testability
func fileHasExtension(path string, suffixes []string) bool {
	ext := filepath.Ext(path)
	if ext != "" && ext[0] == '.' {
		ext = ext[1:]
	}
	for _, s := range suffixes {
		if s != "" && ext == s {
			return true
		}
	}
	return false
}

// locateTemplateFiles ensures that the snapshot has mapped template files
// within the workspace folder.
func (s *snapshot) locateTemplateFiles(ctx context.Context) {
	if len(s.view.Options().TemplateExtensions) == 0 {
		return
	}
	suffixes := s.view.Options().TemplateExtensions

	searched := 0
	filterFunc := s.view.filterFunc()
	err := filepath.WalkDir(s.view.folder.Filename(), func(path string, entry os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if entry.IsDir() {
			return nil
		}
		if fileLimit > 0 && searched > fileLimit {
			return errExhausted
		}
		searched++
		if !fileHasExtension(path, suffixes) {
			return nil
		}
		uri := span.URIFromPath(path)
		if filterFunc(uri) {
			return nil
		}
		// Get the file in order to include it in the snapshot.
		// TODO(golang/go#57558): it is fundamentally broken to track files in this
		// way; we may lose them if configuration or layout changes cause a view to
		// be recreated.
		//
		// Furthermore, this operation must ignore errors, including context
		// cancellation, or risk leaving the snapshot in an undefined state.
		s.ReadFile(ctx, uri)
		return nil
	})
	if err != nil {
		event.Error(ctx, "searching for template files failed", err)
	}
}

func (v *View) contains(uri span.URI) bool {
	// If we've expanded the go dir to a parent directory, consider if the
	// expanded dir contains the uri.
	// TODO(rfindley): should we ignore the root here? It is not provided by the
	// user. It would be better to explicitly consider the set of active modules
	// wherever relevant.
	inGoDir := false
	if source.InDir(v.workingDir().Filename(), v.folder.Filename()) {
		inGoDir = source.InDir(v.workingDir().Filename(), uri.Filename())
	}
	inFolder := source.InDir(v.folder.Filename(), uri.Filename())

	if !inGoDir && !inFolder {
		return false
	}

	return !v.filterFunc()(uri)
}

// filterFunc returns a func that reports whether uri is filtered by the currently configured
// directoryFilters.
func (v *View) filterFunc() func(span.URI) bool {
	filterer := buildFilterer(v.folder.Filename(), v.gomodcache, v.Options())
	return func(uri span.URI) bool {
		// Only filter relative to the configured root directory.
		if source.InDir(v.folder.Filename(), uri.Filename()) {
			return pathExcludedByFilter(strings.TrimPrefix(uri.Filename(), v.folder.Filename()), filterer)
		}
		return false
	}
}

func (v *View) relevantChange(c source.FileModification) bool {
	// If the file is known to the view, the change is relevant.
	if v.knownFile(c.URI) {
		return true
	}
	// The go.work file may not be "known" because we first access it through the
	// session. As a result, treat changes to the view's go.work file as always
	// relevant, even if they are only on-disk changes.
	//
	// TODO(rfindley): Make sure the go.work files are always known
	// to the view.
	if gowork, _ := v.GOWORK(); gowork == c.URI {
		return true
	}

	// Note: CL 219202 filtered out on-disk changes here that were not known to
	// the view, but this introduces a race when changes arrive before the view
	// is initialized (and therefore, before it knows about files). Since that CL
	// had neither test nor associated issue, and cited only emacs behavior, this
	// logic was deleted.

	return v.contains(c.URI)
}

func (v *View) markKnown(uri span.URI) {
	v.knownFilesMu.Lock()
	defer v.knownFilesMu.Unlock()
	if v.knownFiles == nil {
		v.knownFiles = make(map[span.URI]bool)
	}
	v.knownFiles[uri] = true
}

// knownFile reports whether the specified valid URI (or an alias) is known to the view.
func (v *View) knownFile(uri span.URI) bool {
	v.knownFilesMu.Lock()
	defer v.knownFilesMu.Unlock()
	return v.knownFiles[uri]
}

// shutdown releases resources associated with the view, and waits for ongoing
// work to complete.
func (v *View) shutdown() {
	// Cancel the initial workspace load if it is still running.
	v.initCancelFirstAttempt()

	v.snapshotMu.Lock()
	if v.snapshot != nil {
		v.snapshot.cancel()
		v.releaseSnapshot()
		v.destroy(v.snapshot, "View.shutdown")
		v.snapshot = nil
		v.releaseSnapshot = nil
	}
	v.snapshotMu.Unlock()

	v.snapshotWG.Wait()
}

// While go list ./... skips directories starting with '.', '_', or 'testdata',
// gopls may still load them via file queries. Explicitly filter them out.
func (s *snapshot) IgnoredFile(uri span.URI) bool {
	// Fast path: if uri doesn't contain '.', '_', or 'testdata', it is not
	// possible that it is ignored.
	{
		uriStr := string(uri)
		if !strings.Contains(uriStr, ".") && !strings.Contains(uriStr, "_") && !strings.Contains(uriStr, "testdata") {
			return false
		}
	}

	s.ignoreFilterOnce.Do(func() {
		var dirs []string
		if len(s.workspaceModFiles) == 0 {
			for _, entry := range filepath.SplitList(s.view.gopath) {
				dirs = append(dirs, filepath.Join(entry, "src"))
			}
		} else {
			dirs = append(dirs, s.view.gomodcache)
			for m := range s.workspaceModFiles {
				dirs = append(dirs, filepath.Dir(m.Filename()))
			}
		}
		s.ignoreFilter = newIgnoreFilter(dirs)
	})

	return s.ignoreFilter.ignored(uri.Filename())
}

// An ignoreFilter implements go list's exclusion rules via its 'ignored' method.
type ignoreFilter struct {
	prefixes []string // root dirs, ending in filepath.Separator
}

// newIgnoreFilter returns a new ignoreFilter implementing exclusion rules
// relative to the provided directories.
func newIgnoreFilter(dirs []string) *ignoreFilter {
	f := new(ignoreFilter)
	for _, d := range dirs {
		f.prefixes = append(f.prefixes, filepath.Clean(d)+string(filepath.Separator))
	}
	return f
}

func (f *ignoreFilter) ignored(filename string) bool {
	for _, prefix := range f.prefixes {
		if suffix := strings.TrimPrefix(filename, prefix); suffix != filename {
			if checkIgnored(suffix) {
				return true
			}
		}
	}
	return false
}

// checkIgnored implements go list's exclusion rules.
// Quoting “go help list”:
//
//	Directory and file names that begin with "." or "_" are ignored
//	by the go tool, as are directories named "testdata".
func checkIgnored(suffix string) bool {
	// Note: this could be further optimized by writing a HasSegment helper, a
	// segment-boundary respecting variant of strings.Contains.
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

func (v *View) Snapshot() (source.Snapshot, func(), error) {
	return v.getSnapshot()
}

func (v *View) getSnapshot() (*snapshot, func(), error) {
	v.snapshotMu.Lock()
	defer v.snapshotMu.Unlock()
	if v.snapshot == nil {
		return nil, nil, errors.New("view is shutdown")
	}
	return v.snapshot, v.snapshot.Acquire(), nil
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

	s.mu.Lock()
	initialized := s.initialized
	s.mu.Unlock()

	if initialized {
		return
	}

	s.loadWorkspace(ctx, firstAttempt)
	s.collectAllKnownSubdirs(ctx)
}

func (s *snapshot) loadWorkspace(ctx context.Context, firstAttempt bool) (loadErr error) {
	// A failure is retryable if it may have been due to context cancellation,
	// and this is not the initial workspace load (firstAttempt==true).
	//
	// The IWL runs on a detached context with a long (~10m) timeout, so
	// if the context was canceled we consider loading to have failed
	// permanently.
	retryableFailure := func() bool {
		return loadErr != nil && ctx.Err() != nil && !firstAttempt
	}
	defer func() {
		if !retryableFailure() {
			s.mu.Lock()
			s.initialized = true
			s.mu.Unlock()
		}
		if firstAttempt {
			close(s.view.initialWorkspaceLoad)
		}
	}()

	// TODO(rFindley): we should only locate template files on the first attempt,
	// or guard it via a different mechanism.
	s.locateTemplateFiles(ctx)

	// Collect module paths to load by parsing go.mod files. If a module fails to
	// parse, capture the parsing failure as a critical diagnostic.
	var scopes []loadScope                  // scopes to load
	var modDiagnostics []*source.Diagnostic // diagnostics for broken go.mod files
	addError := func(uri span.URI, err error) {
		modDiagnostics = append(modDiagnostics, &source.Diagnostic{
			URI:      uri,
			Severity: protocol.SeverityError,
			Source:   source.ListError,
			Message:  err.Error(),
		})
	}

	// TODO(rfindley): this should be predicated on the s.view.moduleMode().
	// There is no point loading ./... if we have an empty go.work.
	if len(s.workspaceModFiles) > 0 {
		for modURI := range s.workspaceModFiles {
			// Verify that the modfile is valid before trying to load it.
			//
			// TODO(rfindley): now that we no longer need to parse the modfile in
			// order to load scope, we could move these diagnostics to a more general
			// location where we diagnose problems with modfiles or the workspace.
			//
			// Be careful not to add context cancellation errors as critical module
			// errors.
			fh, err := s.ReadFile(ctx, modURI)
			if err != nil {
				if ctx.Err() != nil {
					return ctx.Err()
				}
				addError(modURI, err)
				continue
			}
			parsed, err := s.ParseMod(ctx, fh)
			if err != nil {
				if ctx.Err() != nil {
					return ctx.Err()
				}
				addError(modURI, err)
				continue
			}
			if parsed.File == nil || parsed.File.Module == nil {
				addError(modURI, fmt.Errorf("no module path for %s", modURI))
				continue
			}
			moduleDir := filepath.Dir(modURI.Filename())
			// Previously, we loaded <modulepath>/... for each module path, but that
			// is actually incorrect when the pattern may match packages in more than
			// one module. See golang/go#59458 for more details.
			scopes = append(scopes, moduleLoadScope{dir: moduleDir, modulePath: parsed.File.Module.Mod.Path})
		}
	} else {
		scopes = append(scopes, viewLoadScope("LOAD_VIEW"))
	}

	// If we're loading anything, ensure we also load builtin,
	// since it provides fake definitions (and documentation)
	// for types like int that are used everywhere.
	if len(scopes) > 0 {
		scopes = append(scopes, packageLoadScope("builtin"))
	}
	loadErr = s.load(ctx, true, scopes...)

	if retryableFailure() {
		return loadErr
	}

	var criticalErr *source.CriticalError
	switch {
	case loadErr != nil && ctx.Err() != nil:
		event.Error(ctx, fmt.Sprintf("initial workspace load: %v", loadErr), loadErr)
		criticalErr = &source.CriticalError{
			MainError: loadErr,
		}
	case loadErr != nil:
		event.Error(ctx, "initial workspace load failed", loadErr)
		extractedDiags := s.extractGoCommandErrors(ctx, loadErr)
		criticalErr = &source.CriticalError{
			MainError:   loadErr,
			Diagnostics: append(modDiagnostics, extractedDiags...),
		}
	case len(modDiagnostics) == 1:
		criticalErr = &source.CriticalError{
			MainError:   fmt.Errorf(modDiagnostics[0].Message),
			Diagnostics: modDiagnostics,
		}
	case len(modDiagnostics) > 1:
		criticalErr = &source.CriticalError{
			MainError:   fmt.Errorf("error loading module names"),
			Diagnostics: modDiagnostics,
		}
	}

	// Lock the snapshot when setting the initialized error.
	s.mu.Lock()
	defer s.mu.Unlock()
	s.initializedErr = criticalErr
	return loadErr
}

// invalidateContent invalidates the content of a Go file,
// including any position and type information that depends on it.
//
// invalidateContent returns a non-nil snapshot for the new content, along with
// a callback which the caller must invoke to release that snapshot.
func (v *View) invalidateContent(ctx context.Context, changes map[span.URI]*fileChange, forceReloadMetadata bool) (*snapshot, func()) {
	// Detach the context so that content invalidation cannot be canceled.
	ctx = xcontext.Detach(ctx)

	// This should be the only time we hold the view's snapshot lock for any period of time.
	v.snapshotMu.Lock()
	defer v.snapshotMu.Unlock()

	prevSnapshot, prevReleaseSnapshot := v.snapshot, v.releaseSnapshot

	if prevSnapshot == nil {
		panic("invalidateContent called after shutdown")
	}

	// Cancel all still-running previous requests, since they would be
	// operating on stale data.
	prevSnapshot.cancel()

	// Do not clone a snapshot until its view has finished initializing.
	prevSnapshot.AwaitInitialized(ctx)

	// Save one lease of the cloned snapshot in the view.
	v.snapshot, v.releaseSnapshot = prevSnapshot.clone(ctx, v.baseCtx, changes, forceReloadMetadata)

	prevReleaseSnapshot()
	v.destroy(prevSnapshot, "View.invalidateContent")

	// Return a second lease to the caller.
	return v.snapshot, v.snapshot.Acquire()
}

func (s *Session) getWorkspaceInformation(ctx context.Context, folder span.URI, options *source.Options) (workspaceInformation, error) {
	if err := checkPathCase(folder.Filename()); err != nil {
		return workspaceInformation{}, fmt.Errorf("invalid workspace folder path: %w; check that the casing of the configured workspace folder path agrees with the casing reported by the operating system", err)
	}
	var err error
	var info workspaceInformation
	inv := gocommand.Invocation{
		WorkingDir: folder.Filename(),
		Env:        options.EnvSlice(),
	}
	info.goversion, err = gocommand.GoVersion(ctx, inv, s.gocmdRunner)
	if err != nil {
		return info, err
	}
	info.goversionOutput, err = gocommand.GoVersionOutput(ctx, inv, s.gocmdRunner)
	if err != nil {
		return info, err
	}
	if err := info.load(ctx, folder.Filename(), options.EnvSlice(), s.gocmdRunner); err != nil {
		return info, err
	}
	// The value of GOPACKAGESDRIVER is not returned through the go command.
	gopackagesdriver := os.Getenv("GOPACKAGESDRIVER")
	// A user may also have a gopackagesdriver binary on their machine, which
	// works the same way as setting GOPACKAGESDRIVER.
	tool, _ := exec.LookPath("gopackagesdriver")
	info.hasGopackagesDriver = gopackagesdriver != "off" && (gopackagesdriver != "" || tool != "")

	// filterFunc is the path filter function for this workspace folder. Notably,
	// it is relative to folder (which is specified by the user), not root.
	filterFunc := pathExcludedByFilterFunc(folder.Filename(), info.gomodcache, options)
	info.gomod, err = findWorkspaceModFile(ctx, folder, s, filterFunc)
	if err != nil {
		return info, err
	}

	// Check if the workspace is within any GOPATH directory.
	for _, gp := range filepath.SplitList(info.gopath) {
		if source.InDir(filepath.Join(gp, "src"), folder.Filename()) {
			info.inGOPATH = true
			break
		}
	}
	return info, nil
}

// findWorkspaceModFile searches for a single go.mod file relative to the given
// folder URI, using the following algorithm:
//  1. if there is a go.mod file in a parent directory, return it
//  2. else, if there is exactly one nested module, return it
//  3. else, return ""
func findWorkspaceModFile(ctx context.Context, folderURI span.URI, fs source.FileSource, excludePath func(string) bool) (span.URI, error) {
	folder := folderURI.Filename()
	match, err := findRootPattern(ctx, folder, "go.mod", fs)
	if err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return "", ctxErr
		}
		return "", err
	}
	if match != "" {
		return span.URIFromPath(match), nil
	}

	// ...else we should check if there's exactly one nested module.
	all, err := findModules(folderURI, excludePath, 2)
	if err == errExhausted {
		// Fall-back behavior: if we don't find any modules after searching 10000
		// files, assume there are none.
		event.Log(ctx, fmt.Sprintf("stopped searching for modules after %d files", fileLimit))
		return "", nil
	}
	if err != nil {
		return "", err
	}
	if len(all) == 1 {
		// range to access first element.
		for uri := range all {
			return uri, nil
		}
	}
	return "", nil
}

// workingDir returns the directory from which to run Go commands.
//
// The only case where this should matter is if we've narrowed the workspace to
// a singular nested module. In that case, the go command won't be able to find
// the module unless we tell it the nested directory.
func (v *View) workingDir() span.URI {
	// Note: if gowork is in use, this will default to the workspace folder. In
	// the past, we would instead use the folder containing go.work. This should
	// not make a difference, and in fact may improve go list error messages.
	//
	// TODO(golang/go#57514): eliminate the expandWorkspaceToModule setting
	// entirely.
	if v.Options().ExpandWorkspaceToModule && v.gomod != "" {
		return span.URIFromPath(filepath.Dir(v.gomod.Filename()))
	}
	return v.folder
}

// findRootPattern looks for files with the given basename in dir or any parent
// directory of dir, using the provided FileSource. It returns the first match,
// starting from dir and search parents.
//
// The resulting string is either the file path of a matching file with the
// given basename, or "" if none was found.
func findRootPattern(ctx context.Context, dir, basename string, fs source.FileSource) (string, error) {
	for dir != "" {
		target := filepath.Join(dir, basename)
		exists, err := fileExists(ctx, span.URIFromPath(target), fs)
		if err != nil {
			return "", err // not readable or context cancelled
		}
		if exists {
			return target, nil
		}
		// Trailing separators must be trimmed, otherwise filepath.Split is a noop.
		next, _ := filepath.Split(strings.TrimRight(dir, string(filepath.Separator)))
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

func (v *View) IsGoPrivatePath(target string) bool {
	return globsMatchPath(v.goprivate, target)
}

func (v *View) ModuleUpgrades(modfile span.URI) map[string]string {
	v.moduleUpgradesMu.Lock()
	defer v.moduleUpgradesMu.Unlock()

	upgrades := map[string]string{}
	for mod, ver := range v.moduleUpgrades[modfile] {
		upgrades[mod] = ver
	}
	return upgrades
}

func (v *View) RegisterModuleUpgrades(modfile span.URI, upgrades map[string]string) {
	// Return early if there are no upgrades.
	if len(upgrades) == 0 {
		return
	}

	v.moduleUpgradesMu.Lock()
	defer v.moduleUpgradesMu.Unlock()

	m := v.moduleUpgrades[modfile]
	if m == nil {
		m = make(map[string]string)
		v.moduleUpgrades[modfile] = m
	}
	for mod, ver := range upgrades {
		m[mod] = ver
	}
}

func (v *View) ClearModuleUpgrades(modfile span.URI) {
	v.moduleUpgradesMu.Lock()
	defer v.moduleUpgradesMu.Unlock()

	delete(v.moduleUpgrades, modfile)
}

const maxGovulncheckResultAge = 1 * time.Hour // Invalidate results older than this limit.
var timeNow = time.Now                        // for testing

func (v *View) Vulnerabilities(modfiles ...span.URI) map[span.URI]*govulncheck.Result {
	m := make(map[span.URI]*govulncheck.Result)
	now := timeNow()
	v.vulnsMu.Lock()
	defer v.vulnsMu.Unlock()

	if len(modfiles) == 0 { // empty means all modfiles
		for modfile := range v.vulns {
			modfiles = append(modfiles, modfile)
		}
	}
	for _, modfile := range modfiles {
		vuln := v.vulns[modfile]
		if vuln != nil && now.Sub(vuln.AsOf) > maxGovulncheckResultAge {
			v.vulns[modfile] = nil // same as SetVulnerabilities(modfile, nil)
			vuln = nil
		}
		m[modfile] = vuln
	}
	return m
}

func (v *View) SetVulnerabilities(modfile span.URI, vulns *govulncheck.Result) {
	v.vulnsMu.Lock()
	defer v.vulnsMu.Unlock()

	v.vulns[modfile] = vulns
}

func (v *View) GoVersion() int {
	return v.workspaceInformation.goversion
}

func (v *View) GoVersionString() string {
	return gocommand.ParseGoVersionOutput(v.workspaceInformation.goversionOutput)
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
	// Legacy GOPATH workspace?
	if s.workspaceMode()&moduleMode == 0 {
		return false, nil
	}

	// Explicit -mod flag?
	matches := modFlagRegexp.FindStringSubmatch(s.view.goflags)
	if len(matches) != 0 {
		modFlag := matches[1]
		if modFlag != "" {
			// Don't override an explicit '-mod=vendor' argument.
			// We do want to override '-mod=readonly': it would break various module code lenses,
			// and on 1.16 we know -modfile is available, so we won't mess with go.mod anyway.
			return modFlag == "vendor", nil
		}
	}

	modFile, err := modfile.Parse(modURI.Filename(), modContent, nil)
	if err != nil {
		return false, err
	}

	// No vendor directory?
	// TODO(golang/go#57514): this is wrong if the working dir is not the module
	// root.
	if fi, err := os.Stat(filepath.Join(s.view.workingDir().Filename(), "vendor")); err != nil || !fi.IsDir() {
		return false, nil
	}

	// Vendoring enabled by default by go declaration in go.mod?
	vendorEnabled := modFile.Go != nil && modFile.Go.Version != "" && semver.Compare("v"+modFile.Go.Version, "v1.14") >= 0
	return vendorEnabled, nil
}

// TODO(rfindley): clean up the redundancy of allFilesExcluded,
// pathExcludedByFilterFunc, pathExcludedByFilter, view.filterFunc...
func allFilesExcluded(files []string, filterFunc func(span.URI) bool) bool {
	for _, f := range files {
		uri := span.URIFromPath(f)
		if !filterFunc(uri) {
			return false
		}
	}
	return true
}

func pathExcludedByFilterFunc(folder, gomodcache string, opts *source.Options) func(string) bool {
	filterer := buildFilterer(folder, gomodcache, opts)
	return func(path string) bool {
		return pathExcludedByFilter(path, filterer)
	}
}

// pathExcludedByFilter reports whether the path (relative to the workspace
// folder) should be excluded by the configured directory filters.
//
// TODO(rfindley): passing root and gomodcache here makes it confusing whether
// path should be absolute or relative, and has already caused at least one
// bug.
func pathExcludedByFilter(path string, filterer *source.Filterer) bool {
	path = strings.TrimPrefix(filepath.ToSlash(path), "/")
	return filterer.Disallow(path)
}

func buildFilterer(folder, gomodcache string, opts *source.Options) *source.Filterer {
	filters := opts.DirectoryFilters

	if pref := strings.TrimPrefix(gomodcache, folder); pref != gomodcache {
		modcacheFilter := "-" + strings.TrimPrefix(filepath.ToSlash(pref), "/")
		filters = append(filters, modcacheFilter)
	}
	return source.NewFilterer(filters)
}

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package cache implements the caching layer for gopls.
package cache

import (
	"bytes"
	"context"
	"encoding/json"
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
	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/gopls/internal/govulncheck"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/gopls/internal/span"
	"golang.org/x/tools/internal/bug"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/gocommand"
	"golang.org/x/tools/internal/imports"
	"golang.org/x/tools/internal/xcontext"
)

type View struct {
	id string

	cache       *Cache            // shared cache
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
	//
	// TODO(rfindley): consolidate and/or eliminate redundancy in these fields,
	// which have evolved from different sources over time.
	folder               span.URI // user-specified workspace folder
	rootURI              span.URI // either folder or dir(rootSrc) (TODO: deprecate, in favor of folder+rootSrc)
	rootSrc              span.URI // file providing module information (go.mod or go.work); may be empty
	explicitGowork       span.URI // explicitGowork: if non-empty, a user-specified go.work location (TODO: deprecate)
	workspaceInformation          // grab-bag of Go environment information (TODO: cleanup)

	importsState *importsState

	// moduleUpgrades tracks known upgrades for module paths in each modfile.
	// Each modfile has a map of module name to upgrade version.
	moduleUpgradesMu sync.Mutex
	moduleUpgrades   map[span.URI]map[string]string

	// vulns maps each go.mod file's URI to its known vulnerabilities.
	vulnsMu sync.Mutex
	vulns   map[span.URI]*govulncheck.Result

	// filesByURI maps URIs to the canonical URI for the file it denotes.
	// We also keep a set of candidates for a given basename
	// to reduce the set of pairs that need to be tested for sameness.
	//
	// TODO(rfindley): move this file tracking to the session.
	filesByMu   sync.Mutex
	filesByURI  map[span.URI]span.URI     // key is noncanonical URI (alias)
	filesByBase map[string][]canonicalURI // key is basename

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

type workspaceInformation struct {
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

	// `go env` variables that need to be tracked by gopls.
	//
	// TODO(rfindley): eliminate this in favor of goEnv, or vice-versa.
	environmentVariables

	// goEnv is the `go env` output collected when a view is created.
	// It includes the values of the environment variables above.
	goEnv map[string]string
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

// environmentVariables holds important environment variables captured by a
// call to `go env`.
type environmentVariables struct {
	gocache, gopath, goroot, goprivate, gomodcache string

	// Don't use go111module directly, because we choose to use a different
	// default (auto) on Go 1.16 and later, to avoid spurious errors. Use
	// the workspaceInformation.GO111MODULE method instead.
	go111module string
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

	content, err := modFh.Read()
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
	if o, ok := fh.(source.Overlay); ok {
		if o.Kind() != source.UnknownKind {
			return o.Kind()
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
	// Check if any of the settings that modify our understanding of files have been changed
	if !reflect.DeepEqual(a.Env, b.Env) {
		return false
	}
	if !reflect.DeepEqual(a.DirectoryFilters, b.DirectoryFilters) {
		return false
	}
	if !reflect.DeepEqual(a.StandaloneTags, b.StandaloneTags) {
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
func viewEnv(v *View) string {
	v.optionsMu.Lock()
	env := v.options.EnvSlice()
	buildFlags := append([]string{}, v.options.BuildFlags...)
	v.optionsMu.Unlock()

	var buf bytes.Buffer
	fmt.Fprintf(&buf, `go env for %v
(root %s)
(go version %s)
(valid build configuration = %v)
(build flags: %v)
`,
		v.folder.Filename(),
		v.rootURI.Filename(),
		strings.TrimRight(v.workspaceInformation.goversionOutput, "\n"),
		v.snapshot.ValidBuildConfiguration(),
		buildFlags)

	fullEnv := make(map[string]string)
	for k, v := range v.goEnv {
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
	for k, v := range fullEnv {
		fmt.Fprintf(&buf, "%s=%s\n", k, v)
	}

	return buf.String()
}

func (s *snapshot) RunProcessEnvFunc(ctx context.Context, fn func(*imports.Options) error) error {
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

func (s *snapshot) locateTemplateFiles(ctx context.Context) {
	if len(s.view.Options().TemplateExtensions) == 0 {
		return
	}
	suffixes := s.view.Options().TemplateExtensions

	// The workspace root may have been expanded to a module, but we should apply
	// directory filters based on the configured workspace folder.
	//
	// TODO(rfindley): we should be more principled about paths outside of the
	// workspace folder: do we even consider them? Do we support absolute
	// exclusions? Relative exclusions starting with ..?
	dir := s.workspace.root.Filename()
	relativeTo := s.view.folder.Filename()

	searched := 0
	filterer := buildFilterer(dir, s.view.gomodcache, s.view.Options())
	// Change to WalkDir when we move up to 1.16
	err := filepath.Walk(dir, func(path string, fi os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		relpath := strings.TrimPrefix(path, relativeTo)
		excluded := pathExcludedByFilter(relpath, filterer)
		if fileHasExtension(path, suffixes) && !excluded && !fi.IsDir() {
			k := span.URIFromPath(path)
			_, err := s.GetVersionedFile(ctx, k)
			if err != nil {
				return nil
			}
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
	// TODO(rfindley): should we ignore the root here? It is not provided by the
	// user, and is undefined when go.work is outside the workspace. It would be
	// better to explicitly consider the set of active modules wherever relevant.
	inRoot := source.InDir(v.rootURI.Filename(), uri.Filename())
	inFolder := source.InDir(v.folder.Filename(), uri.Filename())

	if !inRoot && !inFolder {
		return false
	}

	return !v.filterFunc()(uri)
}

// filterFunc returns a func that reports whether uri is filtered by the currently configured
// directoryFilters.
func (v *View) filterFunc() func(span.URI) bool {
	filterer := buildFilterer(v.rootURI.Filename(), v.gomodcache, v.Options())
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
	// The go.work/gopls.mod may not be "known" because we first access it
	// through the session. As a result, treat changes to the view's go.work or
	// gopls.mod file as always relevant, even if they are only on-disk
	// changes.
	// TODO(rstambler): Make sure the go.work/gopls.mod files are always known
	// to the view.
	for _, src := range []workspaceSource{goWorkWorkspace, goplsModWorkspace} {
		if c.URI == uriForSource(v.rootURI, v.explicitGowork, src) {
			return true
		}
	}

	// Note: CL 219202 filtered out on-disk changes here that were not known to
	// the view, but this introduces a race when changes arrive before the view
	// is initialized (and therefore, before it knows about files). Since that CL
	// had neither test nor associated issue, and cited only emacs behavior, this
	// logic was deleted.

	return v.contains(c.URI)
}

// knownFile reports whether the specified valid URI (or an alias) is known to the view.
func (v *View) knownFile(uri span.URI) bool {
	_, known := v.canonicalURI(uri, false)
	return known
}

// TODO(adonovan): opt: eliminate 'filename' optimization. I doubt the
// cost of allocation is significant relative to the
// stat/open/fstat/close operations that follow on Windows.
type canonicalURI struct {
	uri      span.URI
	filename string // = uri.Filename(), an optimization (on Windows)
}

// canonicalURI returns the canonical URI that denotes the same file
// as uri, which may differ due to case insensitivity, unclean paths,
// soft or hard links, and so on.  If no previous alias was found, or
// the file is missing, insert determines whether to make uri the
// canonical representative of the file or to return false.
//
// The cache grows indefinitely without invalidation: file system
// operations may cause two URIs that used to denote the same file to
// no longer to do so. Also, the basename cache grows without bound.
// TODO(adonovan): fix both bugs.
func (v *View) canonicalURI(uri span.URI, insert bool) (span.URI, bool) {
	v.filesByMu.Lock()
	defer v.filesByMu.Unlock()

	// Have we seen this exact URI before?
	if canonical, ok := v.filesByURI[uri]; ok {
		return canonical, true
	}

	// Inspect all candidates with the same lowercase basename.
	// This heuristic is easily defeated by symbolic links to files.
	// Files with some basenames (e.g. doc.go) are very numerous.
	//
	// The set of candidates grows without bound, and incurs a
	// linear sequence of SameFile queries to the file system.
	//
	// It is tempting to fetch the device/inode pair that
	// uniquely identifies a file once, and then compare those
	// pairs, but that would cause us to cache stale file system
	// state (in addition to the filesByURI staleness).
	filename := uri.Filename()
	basename := strings.ToLower(filepath.Base(filename))
	if candidates := v.filesByBase[basename]; candidates != nil {
		if pathStat, _ := os.Stat(filename); pathStat != nil {
			for _, c := range candidates {
				if cStat, _ := os.Stat(c.filename); cStat != nil {
					// On Windows, SameFile is more expensive as it must
					// open the file and use the equivalent of fstat(2).
					if os.SameFile(pathStat, cStat) {
						v.filesByURI[uri] = c.uri
						return c.uri, true
					}
				}
			}
		}
	}

	// No candidates, stat failed, or no candidate matched.
	if insert {
		v.filesByURI[uri] = uri
		v.filesByBase[basename] = append(v.filesByBase[basename], canonicalURI{uri, filename})
	}
	return uri, insert
}

// shutdown releases resources associated with the view, and waits for ongoing
// work to complete.
func (v *View) shutdown() {
	// Cancel the initial workspace load if it is still running.
	v.initCancelFirstAttempt()

	v.snapshotMu.Lock()
	if v.snapshot != nil {
		v.releaseSnapshot()
		v.destroy(v.snapshot, "View.shutdown")
		v.snapshot = nil
		v.releaseSnapshot = nil
	}
	v.snapshotMu.Unlock()

	v.importsState.destroy()
	v.snapshotWG.Wait()
}

func (s *snapshot) IgnoredFile(uri span.URI) bool {
	filename := uri.Filename()
	var prefixes []string
	if len(s.workspace.ActiveModFiles()) == 0 {
		for _, entry := range filepath.SplitList(s.view.gopath) {
			prefixes = append(prefixes, filepath.Join(entry, "src"))
		}
	} else {
		prefixes = append(prefixes, s.view.gomodcache)
		for m := range s.workspace.ActiveModFiles() {
			prefixes = append(prefixes, span.Dir(m).Filename())
		}
	}
	for _, prefix := range prefixes {
		if strings.HasPrefix(filename, prefix) {
			return checkIgnored(filename[len(prefix):])
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
	return v.getSnapshot()
}

func (v *View) getSnapshot() (*snapshot, func()) {
	v.snapshotMu.Lock()
	defer v.snapshotMu.Unlock()
	if v.snapshot == nil {
		panic("getSnapshot called after shutdown")
	}
	return v.snapshot, v.snapshot.Acquire()
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

func (s *snapshot) loadWorkspace(ctx context.Context, firstAttempt bool) {
	defer func() {
		s.mu.Lock()
		s.initialized = true
		s.mu.Unlock()
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

	if len(s.workspace.ActiveModFiles()) > 0 {
		for modURI := range s.workspace.ActiveModFiles() {
			// Be careful not to add context cancellation errors as critical module
			// errors.
			fh, err := s.GetFile(ctx, modURI)
			if err != nil {
				if ctx.Err() == nil {
					addError(modURI, err)
				}
				continue
			}
			parsed, err := s.ParseMod(ctx, fh)
			if err != nil {
				if ctx.Err() == nil {
					addError(modURI, err)
				}
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

	// If we're loading anything, ensure we also load builtin,
	// since it provides fake definitions (and documentation)
	// for types like int that are used everywhere.
	if len(scopes) > 0 {
		scopes = append(scopes, packageLoadScope("builtin"))
	}
	err := s.load(ctx, true, scopes...)

	// If the context is canceled on the first attempt, loading has failed
	// because the go command has timed out--that should be a critical error.
	if err != nil && !firstAttempt && ctx.Err() != nil {
		return
	}

	var criticalErr *source.CriticalError
	switch {
	case err != nil && ctx.Err() != nil:
		event.Error(ctx, fmt.Sprintf("initial workspace load: %v", err), err)
		criticalErr = &source.CriticalError{
			MainError: err,
		}
	case err != nil:
		event.Error(ctx, "initial workspace load failed", err)
		extractedDiags := s.extractGoCommandErrors(ctx, err)
		criticalErr = &source.CriticalError{
			MainError:   err,
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

func (s *Session) getWorkspaceInformation(ctx context.Context, folder span.URI, options *source.Options) (*workspaceInformation, error) {
	if err := checkPathCase(folder.Filename()); err != nil {
		return nil, fmt.Errorf("invalid workspace folder path: %w; check that the casing of the configured workspace folder path agrees with the casing reported by the operating system", err)
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
	goversionOutput, err := gocommand.GoVersionOutput(ctx, inv, s.gocmdRunner)
	if err != nil {
		return nil, err
	}

	// Make sure to get the `go env` before continuing with initialization.
	envVars, env, err := s.getGoEnv(ctx, folder.Filename(), goversion, options.EnvSlice())
	if err != nil {
		return nil, err
	}
	// The value of GOPACKAGESDRIVER is not returned through the go command.
	gopackagesdriver := os.Getenv("GOPACKAGESDRIVER")
	// TODO(rfindley): this looks wrong, or at least overly defensive. If the
	// value of GOPACKAGESDRIVER is not returned from the go command... why do we
	// look it up here?
	for _, s := range env {
		split := strings.SplitN(s, "=", 2)
		if split[0] == "GOPACKAGESDRIVER" {
			bug.Reportf("found GOPACKAGESDRIVER from the go command") // see note above
			gopackagesdriver = split[1]
		}
	}

	// A user may also have a gopackagesdriver binary on their machine, which
	// works the same way as setting GOPACKAGESDRIVER.
	tool, _ := exec.LookPath("gopackagesdriver")
	hasGopackagesDriver := gopackagesdriver != "off" && (gopackagesdriver != "" || tool != "")

	return &workspaceInformation{
		hasGopackagesDriver:  hasGopackagesDriver,
		goversion:            goversion,
		goversionOutput:      goversionOutput,
		environmentVariables: envVars,
		goEnv:                env,
	}, nil
}

// findWorkspaceModuleSource searches for a "module source" relative to the
// given folder URI. A module source is the go.work or go.mod file that
// provides module information.
//
// As a special case, this function returns a module source in a nested
// directory if it finds no other module source, and exactly one nested module.
//
// If no module source is found, it returns "".
func findWorkspaceModuleSource(ctx context.Context, folderURI span.URI, fs source.FileSource, excludePath func(string) bool, experimental bool) (span.URI, error) {
	patterns := []string{"go.work", "go.mod"}
	if experimental {
		patterns = []string{"go.work", "gopls.mod", "go.mod"}
	}
	folder := folderURI.Filename()
	for _, basename := range patterns {
		match, err := findRootPattern(ctx, folder, basename, fs)
		if err != nil {
			if ctxErr := ctx.Err(); ctxErr != nil {
				return "", ctxErr
			}
			return "", err
		}
		if match != "" {
			return span.URIFromPath(match), nil
		}
	}

	// The experimental workspace can handle nested modules at this point...
	if experimental {
		return "", nil
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

// getGoEnv gets the view's various GO* values.
func (s *Session) getGoEnv(ctx context.Context, folder string, goversion int, configEnv []string) (environmentVariables, map[string]string, error) {
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
	// TODO(rfindley): GOWORK is not a property of the session. It may change
	// when a workfile is added or removed.
	//
	// We need to distinguish between GOWORK values that are set by the GOWORK
	// environment variable, and GOWORK values that are computed based on the
	// location of a go.work file in the directory hierarchy.
	args = append(args, "GOWORK")

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
	//
	// TODO(rfindley): consistent with the treatment of go111module, we should
	// provide a wrapper method rather than mutating this value.
	if envVars.gomodcache == "" && envVars.gopath != "" {
		envVars.gomodcache = filepath.Join(filepath.SplitList(envVars.gopath)[0], "pkg/mod")
	}
	return envVars, env, err
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
	matches := modFlagRegexp.FindStringSubmatch(s.view.goEnv["GOFLAGS"])
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
	if fi, err := os.Stat(filepath.Join(s.view.rootURI.Filename(), "vendor")); err != nil || !fi.IsDir() {
		return false, nil
	}

	// Vendoring enabled by default by go declaration in go.mod?
	vendorEnabled := modFile.Go != nil && modFile.Go.Version != "" && semver.Compare("v"+modFile.Go.Version, "v1.14") >= 0
	return vendorEnabled, nil
}

func (v *View) allFilesExcluded(pkg *packages.Package, filterer *source.Filterer) bool {
	folder := filepath.ToSlash(v.folder.Filename())
	for _, f := range pkg.GoFiles {
		f = filepath.ToSlash(f)
		if !strings.HasPrefix(f, folder) {
			return false
		}
		if !pathExcludedByFilter(strings.TrimPrefix(f, folder), filterer) {
			return false
		}
	}
	return true
}

func pathExcludedByFilterFunc(root, gomodcache string, opts *source.Options) func(string) bool {
	filterer := buildFilterer(root, gomodcache, opts)
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

func buildFilterer(root, gomodcache string, opts *source.Options) *source.Filterer {
	// TODO(rfindley): this looks wrong. If gomodcache isn't actually nested
	// under root, this will do the wrong thing.
	gomodcache = strings.TrimPrefix(filepath.ToSlash(strings.TrimPrefix(gomodcache, root)), "/")
	filters := opts.DirectoryFilters
	if gomodcache != "" {
		filters = append(filters, "-"+gomodcache)
	}
	return source.NewFilterer(filters)
}

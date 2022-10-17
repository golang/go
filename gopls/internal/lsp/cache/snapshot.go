// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"io"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"unsafe"

	"golang.org/x/mod/modfile"
	"golang.org/x/mod/module"
	"golang.org/x/mod/semver"
	"golang.org/x/sync/errgroup"
	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/gopls/internal/span"
	"golang.org/x/tools/internal/bug"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/event/tag"
	"golang.org/x/tools/internal/gocommand"
	"golang.org/x/tools/internal/memoize"
	"golang.org/x/tools/internal/packagesinternal"
	"golang.org/x/tools/internal/persistent"
	"golang.org/x/tools/internal/typesinternal"
)

type snapshot struct {
	id   uint64
	view *View

	cancel        func()
	backgroundCtx context.Context

	store *memoize.Store // cache of handles shared by all snapshots

	refcount    sync.WaitGroup // number of references
	destroyedBy *string        // atomically set to non-nil in Destroy once refcount = 0

	// initialized reports whether the snapshot has been initialized. Concurrent
	// initialization is guarded by the view.initializationSema. Each snapshot is
	// initialized at most once: concurrent initialization is guarded by
	// view.initializationSema.
	initialized bool
	// initializedErr holds the last error resulting from initialization. If
	// initialization fails, we only retry when the the workspace modules change,
	// to avoid too many go/packages calls.
	initializedErr *source.CriticalError

	// mu guards all of the maps in the snapshot, as well as the builtin URI.
	mu sync.Mutex

	// builtin pins the AST and package for builtin.go in memory.
	builtin span.URI

	// meta holds loaded metadata.
	//
	// meta is guarded by mu, but the metadataGraph itself is immutable.
	// TODO(rfindley): in many places we hold mu while operating on meta, even
	// though we only need to hold mu while reading the pointer.
	meta *metadataGraph

	// files maps file URIs to their corresponding FileHandles.
	// It may invalidated when a file's content changes.
	files filesMap

	// parsedGoFiles maps a parseKey to the handle of the future result of parsing it.
	parsedGoFiles *persistent.Map // from parseKey to *memoize.Promise[parseGoResult]

	// parseKeysByURI records the set of keys of parsedGoFiles that
	// need to be invalidated for each URI.
	// TODO(adonovan): opt: parseKey = ParseMode + URI, so this could
	// be just a set of ParseModes, or we could loop over AllParseModes.
	parseKeysByURI parseKeysByURIMap

	// symbolizeHandles maps each file URI to a handle for the future
	// result of computing the symbols declared in that file.
	symbolizeHandles *persistent.Map // from span.URI to *memoize.Promise[symbolizeResult]

	// packages maps a packageKey to a *packageHandle.
	// It may be invalidated when a file's content changes.
	//
	// Invariants to preserve:
	//  - packages.Get(id).m.Metadata == meta.metadata[id].Metadata for all ids
	//  - if a package is in packages, then all of its dependencies should also
	//    be in packages, unless there is a missing import
	packages *persistent.Map // from packageKey to *memoize.Promise[*packageHandle]

	// isActivePackageCache maps package ID to the cached value if it is active or not.
	// It may be invalidated when metadata changes or a new file is opened or closed.
	isActivePackageCache isActivePackageCacheMap

	// actions maps an actionKey to the handle for the future
	// result of execution an analysis pass on a package.
	actions *persistent.Map // from actionKey to *actionHandle

	// workspacePackages contains the workspace's packages, which are loaded
	// when the view is created.
	workspacePackages map[PackageID]PackagePath

	// shouldLoad tracks packages that need to be reloaded, mapping a PackageID
	// to the package paths that should be used to reload it
	//
	// When we try to load a package, we clear it from the shouldLoad map
	// regardless of whether the load succeeded, to prevent endless loads.
	shouldLoad map[PackageID][]PackagePath

	// unloadableFiles keeps track of files that we've failed to load.
	unloadableFiles map[span.URI]struct{}

	// parseModHandles keeps track of any parseModHandles for the snapshot.
	// The handles need not refer to only the view's go.mod file.
	parseModHandles *persistent.Map // from span.URI to *memoize.Promise[parseModResult]

	// parseWorkHandles keeps track of any parseWorkHandles for the snapshot.
	// The handles need not refer to only the view's go.work file.
	parseWorkHandles *persistent.Map // from span.URI to *memoize.Promise[parseWorkResult]

	// Preserve go.mod-related handles to avoid garbage-collecting the results
	// of various calls to the go command. The handles need not refer to only
	// the view's go.mod file.
	modTidyHandles *persistent.Map // from span.URI to *memoize.Promise[modTidyResult]
	modWhyHandles  *persistent.Map // from span.URI to *memoize.Promise[modWhyResult]

	workspace *workspace // (not guarded by mu)

	// The cached result of makeWorkspaceDir, created on demand and deleted by Snapshot.Destroy.
	workspaceDir    string
	workspaceDirErr error

	// knownSubdirs is the set of subdirectories in the workspace, used to
	// create glob patterns for file watching.
	knownSubdirs             knownDirsSet
	knownSubdirsPatternCache string
	// unprocessedSubdirChanges are any changes that might affect the set of
	// subdirectories in the workspace. They are not reflected to knownSubdirs
	// during the snapshot cloning step as it can slow down cloning.
	unprocessedSubdirChanges []*fileChange
}

var _ memoize.RefCounted = (*snapshot)(nil) // snapshots are reference-counted

// Acquire prevents the snapshot from being destroyed until the returned function is called.
//
// (s.Acquire().release() could instead be expressed as a pair of
// method calls s.IncRef(); s.DecRef(). The latter has the advantage
// that the DecRefs are fungible and don't require holding anything in
// addition to the refcounted object s, but paradoxically that is also
// an advantage of the current approach, which forces the caller to
// consider the release function at every stage, making a reference
// leak more obvious.)
func (s *snapshot) Acquire() func() {
	type uP = unsafe.Pointer
	if destroyedBy := atomic.LoadPointer((*uP)(uP(&s.destroyedBy))); destroyedBy != nil {
		log.Panicf("%d: acquire() after Destroy(%q)", s.id, *(*string)(destroyedBy))
	}
	s.refcount.Add(1)
	return s.refcount.Done
}

func (s *snapshot) awaitPromise(ctx context.Context, p *memoize.Promise) (interface{}, error) {
	return p.Get(ctx, s)
}

// destroy waits for all leases on the snapshot to expire then releases
// any resources (reference counts and files) associated with it.
// Snapshots being destroyed can be awaited using v.destroyWG.
//
// TODO(adonovan): move this logic into the release function returned
// by Acquire when the reference count becomes zero. (This would cost
// us the destroyedBy debug info, unless we add it to the signature of
// memoize.RefCounted.Acquire.)
//
// The destroyedBy argument is used for debugging.
//
// v.snapshotMu must be held while calling this function, in order to preserve
// the invariants described by the the docstring for v.snapshot.
func (v *View) destroy(s *snapshot, destroyedBy string) {
	v.snapshotWG.Add(1)
	go func() {
		defer v.snapshotWG.Done()
		s.destroy(destroyedBy)
	}()
}

func (s *snapshot) destroy(destroyedBy string) {
	// Wait for all leases to end before commencing destruction.
	s.refcount.Wait()

	// Report bad state as a debugging aid.
	// Not foolproof: another thread could acquire() at this moment.
	type uP = unsafe.Pointer // looking forward to generics...
	if old := atomic.SwapPointer((*uP)(uP(&s.destroyedBy)), uP(&destroyedBy)); old != nil {
		log.Panicf("%d: Destroy(%q) after Destroy(%q)", s.id, destroyedBy, *(*string)(old))
	}

	s.packages.Destroy()
	s.isActivePackageCache.Destroy()
	s.actions.Destroy()
	s.files.Destroy()
	s.parsedGoFiles.Destroy()
	s.parseKeysByURI.Destroy()
	s.knownSubdirs.Destroy()
	s.symbolizeHandles.Destroy()
	s.parseModHandles.Destroy()
	s.parseWorkHandles.Destroy()
	s.modTidyHandles.Destroy()
	s.modWhyHandles.Destroy()

	if s.workspaceDir != "" {
		if err := os.RemoveAll(s.workspaceDir); err != nil {
			event.Error(context.Background(), "cleaning workspace dir", err)
		}
	}
}

func (s *snapshot) ID() uint64 {
	return s.id
}

func (s *snapshot) View() source.View {
	return s.view
}

func (s *snapshot) BackgroundContext() context.Context {
	return s.backgroundCtx
}

func (s *snapshot) FileSet() *token.FileSet {
	return s.view.session.cache.fset
}

func (s *snapshot) ModFiles() []span.URI {
	var uris []span.URI
	for modURI := range s.workspace.getActiveModFiles() {
		uris = append(uris, modURI)
	}
	return uris
}

func (s *snapshot) WorkFile() span.URI {
	return s.workspace.workFile
}

func (s *snapshot) Templates() map[span.URI]source.VersionedFileHandle {
	s.mu.Lock()
	defer s.mu.Unlock()

	tmpls := map[span.URI]source.VersionedFileHandle{}
	s.files.Range(func(k span.URI, fh source.VersionedFileHandle) {
		if s.view.FileKind(fh) == source.Tmpl {
			tmpls[k] = fh
		}
	})
	return tmpls
}

func (s *snapshot) ValidBuildConfiguration() bool {
	// Since we only really understand the `go` command, if the user has a
	// different GOPACKAGESDRIVER, assume that their configuration is valid.
	if s.view.hasGopackagesDriver {
		return true
	}
	// Check if the user is working within a module or if we have found
	// multiple modules in the workspace.
	if len(s.workspace.getActiveModFiles()) > 0 {
		return true
	}
	// The user may have a multiple directories in their GOPATH.
	// Check if the workspace is within any of them.
	for _, gp := range filepath.SplitList(s.view.gopath) {
		if source.InDir(filepath.Join(gp, "src"), s.view.rootURI.Filename()) {
			return true
		}
	}
	return false
}

// workspaceMode describes the way in which the snapshot's workspace should
// be loaded.
func (s *snapshot) workspaceMode() workspaceMode {
	var mode workspaceMode

	// If the view has an invalid configuration, don't build the workspace
	// module.
	validBuildConfiguration := s.ValidBuildConfiguration()
	if !validBuildConfiguration {
		return mode
	}
	// If the view is not in a module and contains no modules, but still has a
	// valid workspace configuration, do not create the workspace module.
	// It could be using GOPATH or a different build system entirely.
	if len(s.workspace.getActiveModFiles()) == 0 && validBuildConfiguration {
		return mode
	}
	mode |= moduleMode
	options := s.view.Options()
	// The -modfile flag is available for Go versions >= 1.14.
	if options.TempModfile && s.view.workspaceInformation.goversion >= 14 {
		mode |= tempModfile
	}
	return mode
}

// config returns the configuration used for the snapshot's interaction with
// the go/packages API. It uses the given working directory.
//
// TODO(rstambler): go/packages requires that we do not provide overlays for
// multiple modules in on config, so buildOverlay needs to filter overlays by
// module.
func (s *snapshot) config(ctx context.Context, inv *gocommand.Invocation) *packages.Config {
	s.view.optionsMu.Lock()
	verboseOutput := s.view.options.VerboseOutput
	s.view.optionsMu.Unlock()

	cfg := &packages.Config{
		Context:    ctx,
		Dir:        inv.WorkingDir,
		Env:        inv.Env,
		BuildFlags: inv.BuildFlags,
		Mode: packages.NeedName |
			packages.NeedFiles |
			packages.NeedCompiledGoFiles |
			packages.NeedImports |
			packages.NeedDeps |
			packages.NeedTypesSizes |
			packages.NeedModule |
			packages.LoadMode(packagesinternal.DepsErrors) |
			packages.LoadMode(packagesinternal.ForTest),
		Fset:    s.FileSet(),
		Overlay: s.buildOverlay(),
		ParseFile: func(*token.FileSet, string, []byte) (*ast.File, error) {
			panic("go/packages must not be used to parse files")
		},
		Logf: func(format string, args ...interface{}) {
			if verboseOutput {
				event.Log(ctx, fmt.Sprintf(format, args...))
			}
		},
		Tests: true,
	}
	packagesinternal.SetModFile(cfg, inv.ModFile)
	packagesinternal.SetModFlag(cfg, inv.ModFlag)
	// We want to type check cgo code if go/types supports it.
	if typesinternal.SetUsesCgo(&types.Config{}) {
		cfg.Mode |= packages.LoadMode(packagesinternal.TypecheckCgo)
	}
	packagesinternal.SetGoCmdRunner(cfg, s.view.session.gocmdRunner)
	return cfg
}

func (s *snapshot) RunGoCommandDirect(ctx context.Context, mode source.InvocationFlags, inv *gocommand.Invocation) (*bytes.Buffer, error) {
	_, inv, cleanup, err := s.goCommandInvocation(ctx, mode, inv)
	if err != nil {
		return nil, err
	}
	defer cleanup()

	return s.view.session.gocmdRunner.Run(ctx, *inv)
}

func (s *snapshot) RunGoCommandPiped(ctx context.Context, mode source.InvocationFlags, inv *gocommand.Invocation, stdout, stderr io.Writer) error {
	_, inv, cleanup, err := s.goCommandInvocation(ctx, mode, inv)
	if err != nil {
		return err
	}
	defer cleanup()
	return s.view.session.gocmdRunner.RunPiped(ctx, *inv, stdout, stderr)
}

func (s *snapshot) RunGoCommands(ctx context.Context, allowNetwork bool, wd string, run func(invoke func(...string) (*bytes.Buffer, error)) error) (bool, []byte, []byte, error) {
	var flags source.InvocationFlags
	if s.workspaceMode()&tempModfile != 0 {
		flags = source.WriteTemporaryModFile
	} else {
		flags = source.Normal
	}
	if allowNetwork {
		flags |= source.AllowNetwork
	}
	tmpURI, inv, cleanup, err := s.goCommandInvocation(ctx, flags, &gocommand.Invocation{WorkingDir: wd})
	if err != nil {
		return false, nil, nil, err
	}
	defer cleanup()
	invoke := func(args ...string) (*bytes.Buffer, error) {
		inv.Verb = args[0]
		inv.Args = args[1:]
		return s.view.session.gocmdRunner.Run(ctx, *inv)
	}
	if err := run(invoke); err != nil {
		return false, nil, nil, err
	}
	if flags.Mode() != source.WriteTemporaryModFile {
		return false, nil, nil, nil
	}
	var modBytes, sumBytes []byte
	modBytes, err = ioutil.ReadFile(tmpURI.Filename())
	if err != nil && !os.IsNotExist(err) {
		return false, nil, nil, err
	}
	sumBytes, err = ioutil.ReadFile(strings.TrimSuffix(tmpURI.Filename(), ".mod") + ".sum")
	if err != nil && !os.IsNotExist(err) {
		return false, nil, nil, err
	}
	return true, modBytes, sumBytes, nil
}

// goCommandInvocation populates inv with configuration for running go commands on the snapshot.
//
// TODO(rfindley): refactor this function to compose the required configuration
// explicitly, rather than implicitly deriving it from flags and inv.
//
// TODO(adonovan): simplify cleanup mechanism. It's hard to see, but
// it used only after call to tempModFile. Clarify that it is only
// non-nil on success.
func (s *snapshot) goCommandInvocation(ctx context.Context, flags source.InvocationFlags, inv *gocommand.Invocation) (tmpURI span.URI, updatedInv *gocommand.Invocation, cleanup func(), err error) {
	s.view.optionsMu.Lock()
	allowModfileModificationOption := s.view.options.AllowModfileModifications
	allowNetworkOption := s.view.options.AllowImplicitNetworkAccess

	// TODO(rfindley): this is very hard to follow, and may not even be doing the
	// right thing: should inv.Env really trample view.options? Do we ever invoke
	// this with a non-empty inv.Env?
	//
	// We should refactor to make it clearer that the correct env is being used.
	inv.Env = append(append(append(os.Environ(), s.view.options.EnvSlice()...), inv.Env...), "GO111MODULE="+s.view.effectiveGo111Module)
	inv.BuildFlags = append([]string{}, s.view.options.BuildFlags...)
	s.view.optionsMu.Unlock()
	cleanup = func() {} // fallback

	// All logic below is for module mode.
	if s.workspaceMode()&moduleMode == 0 {
		return "", inv, cleanup, nil
	}

	mode, allowNetwork := flags.Mode(), flags.AllowNetwork()
	if !allowNetwork && !allowNetworkOption {
		inv.Env = append(inv.Env, "GOPROXY=off")
	}

	// What follows is rather complicated logic for how to actually run the go
	// command. A word of warning: this is the result of various incremental
	// features added to gopls, and varying behavior of the Go command across Go
	// versions. It can surely be cleaned up significantly, but tread carefully.
	//
	// Roughly speaking we need to resolve four things:
	//  - the working directory.
	//  - the -mod flag
	//  - the -modfile flag
	//
	// These are dependent on a number of factors: whether we need to run in a
	// synthetic workspace, whether flags are supported at the current go
	// version, and what we're actually trying to achieve (the
	// source.InvocationFlags).

	var modURI span.URI
	// Select the module context to use.
	// If we're type checking, we need to use the workspace context, meaning
	// the main (workspace) module. Otherwise, we should use the module for
	// the passed-in working dir.
	if mode == source.LoadWorkspace {
		switch s.workspace.moduleSource {
		case legacyWorkspace:
			for m := range s.workspace.getActiveModFiles() { // range to access the only element
				modURI = m
			}
		case goWorkWorkspace:
			if s.view.goversion >= 18 {
				break
			}
			// Before go 1.18, the Go command did not natively support go.work files,
			// so we 'fake' them with a workspace module.
			fallthrough
		case fileSystemWorkspace, goplsModWorkspace:
			var tmpDir span.URI
			var err error
			tmpDir, err = s.getWorkspaceDir(ctx)
			if err != nil {
				return "", nil, cleanup, err
			}
			inv.WorkingDir = tmpDir.Filename()
			modURI = span.URIFromPath(filepath.Join(tmpDir.Filename(), "go.mod"))
		}
	} else {
		modURI = s.GoModForFile(span.URIFromPath(inv.WorkingDir))
	}

	var modContent []byte
	if modURI != "" {
		modFH, err := s.GetFile(ctx, modURI)
		if err != nil {
			return "", nil, cleanup, err
		}
		modContent, err = modFH.Read()
		if err != nil {
			return "", nil, cleanup, err
		}
	}

	// TODO(rfindley): in the case of go.work mode, modURI is empty and we fall
	// back on the default behavior of vendorEnabled with an empty modURI. Figure
	// out what is correct here and implement it explicitly.
	vendorEnabled, err := s.vendorEnabled(ctx, modURI, modContent)
	if err != nil {
		return "", nil, cleanup, err
	}

	mutableModFlag := ""
	// If the mod flag isn't set, populate it based on the mode and workspace.
	if inv.ModFlag == "" {
		if s.view.goversion >= 16 {
			mutableModFlag = "mod"
		}

		switch mode {
		case source.LoadWorkspace, source.Normal:
			if vendorEnabled {
				inv.ModFlag = "vendor"
			} else if !allowModfileModificationOption {
				inv.ModFlag = "readonly"
			} else {
				inv.ModFlag = mutableModFlag
			}
		case source.WriteTemporaryModFile:
			inv.ModFlag = mutableModFlag
			// -mod must be readonly when using go.work files - see issue #48941
			inv.Env = append(inv.Env, "GOWORK=off")
		}
	}

	// Only use a temp mod file if the modfile can actually be mutated.
	needTempMod := inv.ModFlag == mutableModFlag
	useTempMod := s.workspaceMode()&tempModfile != 0
	if needTempMod && !useTempMod {
		return "", nil, cleanup, source.ErrTmpModfileUnsupported
	}

	// We should use -modfile if:
	//  - the workspace mode supports it
	//  - we're using a go.work file on go1.18+, or we need a temp mod file (for
	//    example, if running go mod tidy in a go.work workspace)
	//
	// TODO(rfindley): this is very hard to follow. Refactor.
	useWorkFile := !needTempMod && s.workspace.moduleSource == goWorkWorkspace && s.view.goversion >= 18
	if useWorkFile {
		// Since we're running in the workspace root, the go command will resolve GOWORK automatically.
	} else if useTempMod {
		if modURI == "" {
			return "", nil, cleanup, fmt.Errorf("no go.mod file found in %s", inv.WorkingDir)
		}
		modFH, err := s.GetFile(ctx, modURI)
		if err != nil {
			return "", nil, cleanup, err
		}
		// Use the go.sum if it happens to be available.
		gosum := s.goSum(ctx, modURI)
		tmpURI, cleanup, err = tempModFile(modFH, gosum)
		if err != nil {
			return "", nil, cleanup, err
		}
		inv.ModFile = tmpURI.Filename()
	}

	return tmpURI, inv, cleanup, nil
}

// usesWorkspaceDir reports whether the snapshot should use a synthetic
// workspace directory for running workspace go commands such as go list.
//
// TODO(rfindley): this logic is duplicated with goCommandInvocation. Clean up
// the latter, and deduplicate.
func (s *snapshot) usesWorkspaceDir() bool {
	switch s.workspace.moduleSource {
	case legacyWorkspace:
		return false
	case goWorkWorkspace:
		if s.view.goversion >= 18 {
			return false
		}
		// Before go 1.18, the Go command did not natively support go.work files,
		// so we 'fake' them with a workspace module.
	}
	return true
}

func (s *snapshot) buildOverlay() map[string][]byte {
	s.mu.Lock()
	defer s.mu.Unlock()

	overlays := make(map[string][]byte)
	s.files.Range(func(uri span.URI, fh source.VersionedFileHandle) {
		overlay, ok := fh.(*overlay)
		if !ok {
			return
		}
		if overlay.saved {
			return
		}
		// TODO(rstambler): Make sure not to send overlays outside of the current view.
		overlays[uri.Filename()] = overlay.text
	})
	return overlays
}

func (s *snapshot) PackagesForFile(ctx context.Context, uri span.URI, mode source.TypecheckMode, includeTestVariants bool) ([]source.Package, error) {
	ctx = event.Label(ctx, tag.URI.Of(uri))

	phs, err := s.packageHandlesForFile(ctx, uri, mode, includeTestVariants)
	if err != nil {
		return nil, err
	}
	var pkgs []source.Package
	for _, ph := range phs {
		pkg, err := ph.await(ctx, s)
		if err != nil {
			return nil, err
		}
		pkgs = append(pkgs, pkg)
	}
	return pkgs, nil
}

func (s *snapshot) PackageForFile(ctx context.Context, uri span.URI, mode source.TypecheckMode, pkgPolicy source.PackageFilter) (source.Package, error) {
	ctx = event.Label(ctx, tag.URI.Of(uri))

	phs, err := s.packageHandlesForFile(ctx, uri, mode, false)
	if err != nil {
		return nil, err
	}

	if len(phs) < 1 {
		return nil, fmt.Errorf("no packages")
	}

	ph := phs[0]
	for _, handle := range phs[1:] {
		switch pkgPolicy {
		case source.WidestPackage:
			if ph == nil || len(handle.CompiledGoFiles()) > len(ph.CompiledGoFiles()) {
				ph = handle
			}
		case source.NarrowestPackage:
			if ph == nil || len(handle.CompiledGoFiles()) < len(ph.CompiledGoFiles()) {
				ph = handle
			}
		}
	}
	if ph == nil {
		return nil, fmt.Errorf("no packages in input")
	}

	return ph.await(ctx, s)
}

func (s *snapshot) packageHandlesForFile(ctx context.Context, uri span.URI, mode source.TypecheckMode, withIntermediateTestVariants bool) ([]*packageHandle, error) {
	// TODO(rfindley): why can't/shouldn't we awaitLoaded here? It seems that if
	// we ask for package handles for a file, we should wait for pending loads.
	// Else we will reload orphaned files before the initial load completes.

	// Check if we should reload metadata for the file. We don't invalidate IDs
	// (though we should), so the IDs will be a better source of truth than the
	// metadata. If there are no IDs for the file, then we should also reload.
	fh, err := s.GetFile(ctx, uri)
	if err != nil {
		return nil, err
	}
	if kind := s.view.FileKind(fh); kind != source.Go {
		return nil, fmt.Errorf("no packages for non-Go file %s (%v)", uri, kind)
	}
	knownIDs, err := s.getOrLoadIDsForURI(ctx, uri)
	if err != nil {
		return nil, err
	}

	var phs []*packageHandle
	for _, id := range knownIDs {
		// Filter out any intermediate test variants. We typically aren't
		// interested in these packages for file= style queries.
		if m := s.getMetadata(id); m != nil && m.IsIntermediateTestVariant() && !withIntermediateTestVariants {
			continue
		}
		parseMode := source.ParseFull
		if mode == source.TypecheckWorkspace {
			parseMode = s.workspaceParseMode(id)
		}

		ph, err := s.buildPackageHandle(ctx, id, parseMode)
		if err != nil {
			return nil, err
		}
		phs = append(phs, ph)
	}
	return phs, nil
}

// getOrLoadIDsForURI returns package IDs associated with the file uri. If no
// such packages exist or if they are known to be stale, it reloads the file.
//
// If experimentalUseInvalidMetadata is set, this function may return package
// IDs with invalid metadata.
func (s *snapshot) getOrLoadIDsForURI(ctx context.Context, uri span.URI) ([]PackageID, error) {
	useInvalidMetadata := s.useInvalidMetadata()

	s.mu.Lock()

	// Start with the set of package associations derived from the last load.
	ids := s.meta.ids[uri]

	hasValidID := false // whether we have any valid package metadata containing uri
	shouldLoad := false // whether any packages containing uri are marked 'shouldLoad'
	for _, id := range ids {
		// TODO(rfindley): remove the defensiveness here. s.meta.metadata[id] must
		// exist.
		if m, ok := s.meta.metadata[id]; ok && m.Valid {
			hasValidID = true
		}
		if len(s.shouldLoad[id]) > 0 {
			shouldLoad = true
		}
	}

	// Check if uri is known to be unloadable.
	//
	// TODO(rfindley): shouldn't we also mark uri as unloadable if the load below
	// fails? Otherwise we endlessly load files with no packages.
	_, unloadable := s.unloadableFiles[uri]

	s.mu.Unlock()

	// Special case: if experimentalUseInvalidMetadata is set and we have any
	// ids, just return them.
	//
	// This is arguably wrong: if the metadata is invalid we should try reloading
	// it. However, this was the pre-existing behavior, and
	// experimentalUseInvalidMetadata will be removed in a future release.
	if !shouldLoad && useInvalidMetadata && len(ids) > 0 {
		return ids, nil
	}

	// Reload if loading is likely to improve the package associations for uri:
	//  - uri is not contained in any valid packages
	//  - ...or one of the packages containing uri is marked 'shouldLoad'
	//  - ...but uri is not unloadable
	if (shouldLoad || !hasValidID) && !unloadable {
		scope := fileLoadScope(uri)
		err := s.load(ctx, false, scope)

		// Guard against failed loads due to context cancellation.
		//
		// Return the context error here as the current operation is no longer
		// valid.
		if ctxErr := ctx.Err(); ctxErr != nil {
			return nil, ctxErr
		}

		// We must clear scopes after loading.
		//
		// TODO(rfindley): unlike reloadWorkspace, this is simply marking loaded
		// packages as loaded. We could do this from snapshot.load and avoid
		// raciness.
		s.clearShouldLoad(scope)

		// Don't return an error here, as we may still return stale IDs.
		// Furthermore, the result of getOrLoadIDsForURI should be consistent upon
		// subsequent calls, even if the file is marked as unloadable.
		if err != nil && !errors.Is(err, errNoPackages) {
			event.Error(ctx, "getOrLoadIDsForURI", err)
		}
	}

	s.mu.Lock()
	ids = s.meta.ids[uri]
	if !useInvalidMetadata {
		var validIDs []PackageID
		for _, id := range ids {
			// TODO(rfindley): remove the defensiveness here as well.
			if m, ok := s.meta.metadata[id]; ok && m.Valid {
				validIDs = append(validIDs, id)
			}
		}
		ids = validIDs
	}
	s.mu.Unlock()

	return ids, nil
}

// Only use invalid metadata for Go versions >= 1.13. Go 1.12 and below has
// issues with overlays that will cause confusing error messages if we reuse
// old metadata.
func (s *snapshot) useInvalidMetadata() bool {
	return s.view.goversion >= 13 && s.view.Options().ExperimentalUseInvalidMetadata
}

func (s *snapshot) GetReverseDependencies(ctx context.Context, id string) ([]source.Package, error) {
	if err := s.awaitLoaded(ctx); err != nil {
		return nil, err
	}
	s.mu.Lock()
	meta := s.meta
	s.mu.Unlock()
	ids := meta.reverseTransitiveClosure(s.useInvalidMetadata(), PackageID(id))

	// Make sure to delete the original package ID from the map.
	delete(ids, PackageID(id))

	var pkgs []source.Package
	for id := range ids {
		pkg, err := s.checkedPackage(ctx, id, s.workspaceParseMode(id))
		if err != nil {
			return nil, err
		}
		pkgs = append(pkgs, pkg)
	}
	return pkgs, nil
}

func (s *snapshot) checkedPackage(ctx context.Context, id PackageID, mode source.ParseMode) (*pkg, error) {
	ph, err := s.buildPackageHandle(ctx, id, mode)
	if err != nil {
		return nil, err
	}
	return ph.await(ctx, s)
}

func (s *snapshot) getImportedBy(id PackageID) []PackageID {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.meta.importedBy[id]
}

func (s *snapshot) workspacePackageIDs() (ids []PackageID) {
	s.mu.Lock()
	defer s.mu.Unlock()

	for id := range s.workspacePackages {
		ids = append(ids, id)
	}
	return ids
}

func (s *snapshot) activePackageIDs() (ids []PackageID) {
	if s.view.Options().MemoryMode == source.ModeNormal {
		return s.workspacePackageIDs()
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	for id := range s.workspacePackages {
		if s.isActiveLocked(id) {
			ids = append(ids, id)
		}
	}
	return ids
}

func (s *snapshot) isActiveLocked(id PackageID) (active bool) {
	if seen, ok := s.isActivePackageCache.Get(id); ok {
		return seen
	}
	defer func() {
		s.isActivePackageCache.Set(id, active)
	}()
	m, ok := s.meta.metadata[id]
	if !ok {
		return false
	}
	for _, cgf := range m.CompiledGoFiles {
		if s.isOpenLocked(cgf) {
			return true
		}
	}
	// TODO(rfindley): it looks incorrect that we don't also check GoFiles here.
	// If a CGo file is open, we want to consider the package active.
	for _, dep := range m.Imports {
		if s.isActiveLocked(dep) {
			return true
		}
	}
	return false
}

func (s *snapshot) resetIsActivePackageLocked() {
	s.isActivePackageCache.Destroy()
	s.isActivePackageCache = newIsActivePackageCacheMap()
}

const fileExtensions = "go,mod,sum,work"

func (s *snapshot) fileWatchingGlobPatterns(ctx context.Context) map[string]struct{} {
	extensions := fileExtensions
	for _, ext := range s.View().Options().TemplateExtensions {
		extensions += "," + ext
	}
	// Work-around microsoft/vscode#100870 by making sure that we are,
	// at least, watching the user's entire workspace. This will still be
	// applied to every folder in the workspace.
	patterns := map[string]struct{}{
		fmt.Sprintf("**/*.{%s}", extensions): {},
	}

	if s.view.explicitGowork != "" {
		patterns[s.view.explicitGowork.Filename()] = struct{}{}
	}

	// Add a pattern for each Go module in the workspace that is not within the view.
	dirs := s.workspace.dirs(ctx, s)
	for _, dir := range dirs {
		dirName := dir.Filename()

		// If the directory is within the view's folder, we're already watching
		// it with the pattern above.
		if source.InDir(s.view.folder.Filename(), dirName) {
			continue
		}
		// TODO(rstambler): If microsoft/vscode#3025 is resolved before
		// microsoft/vscode#101042, we will need a work-around for Windows
		// drive letter casing.
		patterns[fmt.Sprintf("%s/**/*.{%s}", dirName, extensions)] = struct{}{}
	}

	// Some clients do not send notifications for changes to directories that
	// contain Go code (golang/go#42348). To handle this, explicitly watch all
	// of the directories in the workspace. We find them by adding the
	// directories of every file in the snapshot's workspace directories.
	// There may be thousands.
	if pattern := s.getKnownSubdirsPattern(dirs); pattern != "" {
		patterns[pattern] = struct{}{}
	}

	return patterns
}

func (s *snapshot) getKnownSubdirsPattern(wsDirs []span.URI) string {
	s.mu.Lock()
	defer s.mu.Unlock()

	// First, process any pending changes and update the set of known
	// subdirectories.
	// It may change list of known subdirs and therefore invalidate the cache.
	s.applyKnownSubdirsChangesLocked(wsDirs)

	if s.knownSubdirsPatternCache == "" {
		var builder strings.Builder
		s.knownSubdirs.Range(func(uri span.URI) {
			if builder.Len() == 0 {
				builder.WriteString("{")
			} else {
				builder.WriteString(",")
			}
			builder.WriteString(uri.Filename())
		})
		if builder.Len() > 0 {
			builder.WriteString("}")
			s.knownSubdirsPatternCache = builder.String()
		}
	}

	return s.knownSubdirsPatternCache
}

// collectAllKnownSubdirs collects all of the subdirectories within the
// snapshot's workspace directories. None of the workspace directories are
// included.
func (s *snapshot) collectAllKnownSubdirs(ctx context.Context) {
	dirs := s.workspace.dirs(ctx, s)

	s.mu.Lock()
	defer s.mu.Unlock()

	s.knownSubdirs.Destroy()
	s.knownSubdirs = newKnownDirsSet()
	s.knownSubdirsPatternCache = ""
	s.files.Range(func(uri span.URI, fh source.VersionedFileHandle) {
		s.addKnownSubdirLocked(uri, dirs)
	})
}

func (s *snapshot) getKnownSubdirs(wsDirs []span.URI) knownDirsSet {
	s.mu.Lock()
	defer s.mu.Unlock()

	// First, process any pending changes and update the set of known
	// subdirectories.
	s.applyKnownSubdirsChangesLocked(wsDirs)

	return s.knownSubdirs.Clone()
}

func (s *snapshot) applyKnownSubdirsChangesLocked(wsDirs []span.URI) {
	for _, c := range s.unprocessedSubdirChanges {
		if c.isUnchanged {
			continue
		}
		if !c.exists {
			s.removeKnownSubdirLocked(c.fileHandle.URI())
		} else {
			s.addKnownSubdirLocked(c.fileHandle.URI(), wsDirs)
		}
	}
	s.unprocessedSubdirChanges = nil
}

func (s *snapshot) addKnownSubdirLocked(uri span.URI, dirs []span.URI) {
	dir := filepath.Dir(uri.Filename())
	// First check if the directory is already known, because then we can
	// return early.
	if s.knownSubdirs.Contains(span.URIFromPath(dir)) {
		return
	}
	var matched span.URI
	for _, wsDir := range dirs {
		if source.InDir(wsDir.Filename(), dir) {
			matched = wsDir
			break
		}
	}
	// Don't watch any directory outside of the workspace directories.
	if matched == "" {
		return
	}
	for {
		if dir == "" || dir == matched.Filename() {
			break
		}
		uri := span.URIFromPath(dir)
		if s.knownSubdirs.Contains(uri) {
			break
		}
		s.knownSubdirs.Insert(uri)
		dir = filepath.Dir(dir)
		s.knownSubdirsPatternCache = ""
	}
}

func (s *snapshot) removeKnownSubdirLocked(uri span.URI) {
	dir := filepath.Dir(uri.Filename())
	for dir != "" {
		uri := span.URIFromPath(dir)
		if !s.knownSubdirs.Contains(uri) {
			break
		}
		if info, _ := os.Stat(dir); info == nil {
			s.knownSubdirs.Remove(uri)
			s.knownSubdirsPatternCache = ""
		}
		dir = filepath.Dir(dir)
	}
}

// knownFilesInDir returns the files known to the given snapshot that are in
// the given directory. It does not respect symlinks.
func (s *snapshot) knownFilesInDir(ctx context.Context, dir span.URI) []span.URI {
	var files []span.URI
	s.mu.Lock()
	defer s.mu.Unlock()

	s.files.Range(func(uri span.URI, fh source.VersionedFileHandle) {
		if source.InDir(dir.Filename(), uri.Filename()) {
			files = append(files, uri)
		}
	})
	return files
}

func (s *snapshot) ActivePackages(ctx context.Context) ([]source.Package, error) {
	phs, err := s.activePackageHandles(ctx)
	if err != nil {
		return nil, err
	}
	var pkgs []source.Package
	for _, ph := range phs {
		pkg, err := ph.await(ctx, s)
		if err != nil {
			return nil, err
		}
		pkgs = append(pkgs, pkg)
	}
	return pkgs, nil
}

func (s *snapshot) activePackageHandles(ctx context.Context) ([]*packageHandle, error) {
	if err := s.awaitLoaded(ctx); err != nil {
		return nil, err
	}
	var phs []*packageHandle
	for _, pkgID := range s.activePackageIDs() {
		ph, err := s.buildPackageHandle(ctx, pkgID, s.workspaceParseMode(pkgID))
		if err != nil {
			return nil, err
		}
		phs = append(phs, ph)
	}
	return phs, nil
}

// Symbols extracts and returns the symbols for each file in all the snapshot's views.
func (s *snapshot) Symbols(ctx context.Context) map[span.URI][]source.Symbol {
	var (
		group    errgroup.Group
		nprocs   = 2 * runtime.GOMAXPROCS(-1)  // symbolize is a mix of I/O and CPU
		iolimit  = make(chan struct{}, nprocs) // I/O limiting counting semaphore
		resultMu sync.Mutex
		result   = make(map[span.URI][]source.Symbol)
	)
	s.files.Range(func(uri span.URI, f source.VersionedFileHandle) {
		if s.View().FileKind(f) != source.Go {
			return // workspace symbols currently supports only Go files.
		}

		// TODO(adonovan): upgrade errgroup and use group.SetLimit(nprocs).
		iolimit <- struct{}{} // acquire token
		group.Go(func() error {
			defer func() { <-iolimit }() // release token
			symbols, err := s.symbolize(ctx, f)
			if err != nil {
				return err
			}
			resultMu.Lock()
			result[uri] = symbols
			resultMu.Unlock()
			return nil
		})
	})
	// Keep going on errors, but log the first failure.
	// Partial results are better than no symbol results.
	if err := group.Wait(); err != nil {
		event.Error(ctx, "getting snapshot symbols", err)
	}
	return result
}

func (s *snapshot) MetadataForFile(ctx context.Context, uri span.URI) ([]source.Metadata, error) {
	knownIDs, err := s.getOrLoadIDsForURI(ctx, uri)
	if err != nil {
		return nil, err
	}
	var mds []source.Metadata
	for _, id := range knownIDs {
		md := s.getMetadata(id)
		// TODO(rfindley): knownIDs and metadata should be in sync, but existing
		// code is defensive of nil metadata.
		if md != nil {
			mds = append(mds, md)
		}
	}
	return mds, nil
}

func (s *snapshot) KnownPackages(ctx context.Context) ([]source.Package, error) {
	if err := s.awaitLoaded(ctx); err != nil {
		return nil, err
	}

	// The WorkspaceSymbols implementation relies on this function returning
	// workspace packages first.
	ids := s.workspacePackageIDs()
	s.mu.Lock()
	for id := range s.meta.metadata {
		if _, ok := s.workspacePackages[id]; ok {
			continue
		}
		ids = append(ids, id)
	}
	s.mu.Unlock()

	var pkgs []source.Package
	for _, id := range ids {
		pkg, err := s.checkedPackage(ctx, id, s.workspaceParseMode(id))
		if err != nil {
			return nil, err
		}
		pkgs = append(pkgs, pkg)
	}
	return pkgs, nil
}

func (s *snapshot) AllValidMetadata(ctx context.Context) ([]source.Metadata, error) {
	if err := s.awaitLoaded(ctx); err != nil {
		return nil, err
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	var meta []source.Metadata
	for _, m := range s.meta.metadata {
		if m.Valid {
			meta = append(meta, m)
		}
	}
	return meta, nil
}

func (s *snapshot) WorkspacePackageByID(ctx context.Context, id string) (source.Package, error) {
	packageID := PackageID(id)
	return s.checkedPackage(ctx, packageID, s.workspaceParseMode(packageID))
}

func (s *snapshot) CachedImportPaths(ctx context.Context) (map[string]source.Package, error) {
	// Don't reload workspace package metadata.
	// This function is meant to only return currently cached information.
	s.AwaitInitialized(ctx)

	s.mu.Lock()
	defer s.mu.Unlock()

	results := map[string]source.Package{}
	s.packages.Range(func(_, v interface{}) {
		cachedPkg, err := v.(*packageHandle).cached()
		if err != nil {
			return
		}
		for pkgPath, newPkg := range cachedPkg.depsByPkgPath {
			if oldPkg, ok := results[string(pkgPath)]; ok {
				// Using the same trick as NarrowestPackage, prefer non-variants.
				if len(newPkg.compiledGoFiles) < len(oldPkg.(*pkg).compiledGoFiles) {
					results[string(pkgPath)] = newPkg
				}
			} else {
				results[string(pkgPath)] = newPkg
			}
		}
	})
	return results, nil
}

func (s *snapshot) GoModForFile(uri span.URI) span.URI {
	return moduleForURI(s.workspace.activeModFiles, uri)
}

func moduleForURI(modFiles map[span.URI]struct{}, uri span.URI) span.URI {
	var match span.URI
	for modURI := range modFiles {
		if !source.InDir(dirURI(modURI).Filename(), uri.Filename()) {
			continue
		}
		if len(modURI) > len(match) {
			match = modURI
		}
	}
	return match
}

func (s *snapshot) getMetadata(id PackageID) *KnownMetadata {
	s.mu.Lock()
	defer s.mu.Unlock()

	return s.meta.metadata[id]
}

// clearShouldLoad clears package IDs that no longer need to be reloaded after
// scopes has been loaded.
func (s *snapshot) clearShouldLoad(scopes ...loadScope) {
	s.mu.Lock()
	defer s.mu.Unlock()

	for _, scope := range scopes {
		switch scope := scope.(type) {
		case packageLoadScope:
			scopePath := PackagePath(scope)
			var toDelete []PackageID
			for id, pkgPaths := range s.shouldLoad {
				for _, pkgPath := range pkgPaths {
					if pkgPath == scopePath {
						toDelete = append(toDelete, id)
					}
				}
			}
			for _, id := range toDelete {
				delete(s.shouldLoad, id)
			}
		case fileLoadScope:
			uri := span.URI(scope)
			ids := s.meta.ids[uri]
			for _, id := range ids {
				delete(s.shouldLoad, id)
			}
		}
	}
}

// noValidMetadataForURILocked reports whether there is any valid metadata for
// the given URI.
func (s *snapshot) noValidMetadataForURILocked(uri span.URI) bool {
	ids, ok := s.meta.ids[uri]
	if !ok {
		return true
	}
	for _, id := range ids {
		if m, ok := s.meta.metadata[id]; ok && m.Valid {
			return false
		}
	}
	return true
}

func (s *snapshot) isWorkspacePackage(id PackageID) bool {
	s.mu.Lock()
	defer s.mu.Unlock()

	_, ok := s.workspacePackages[id]
	return ok
}

func (s *snapshot) FindFile(uri span.URI) source.VersionedFileHandle {
	f := s.view.getFile(uri)

	s.mu.Lock()
	defer s.mu.Unlock()

	result, _ := s.files.Get(f.URI())
	return result
}

// GetVersionedFile returns a File for the given URI. If the file is unknown it
// is added to the managed set.
//
// GetVersionedFile succeeds even if the file does not exist. A non-nil error return
// indicates some type of internal error, for example if ctx is cancelled.
func (s *snapshot) GetVersionedFile(ctx context.Context, uri span.URI) (source.VersionedFileHandle, error) {
	f := s.view.getFile(uri)

	s.mu.Lock()
	defer s.mu.Unlock()
	return s.getFileLocked(ctx, f)
}

// GetFile implements the fileSource interface by wrapping GetVersionedFile.
func (s *snapshot) GetFile(ctx context.Context, uri span.URI) (source.FileHandle, error) {
	return s.GetVersionedFile(ctx, uri)
}

func (s *snapshot) getFileLocked(ctx context.Context, f *fileBase) (source.VersionedFileHandle, error) {
	if fh, ok := s.files.Get(f.URI()); ok {
		return fh, nil
	}

	fh, err := s.view.session.cache.getFile(ctx, f.URI()) // read the file
	if err != nil {
		return nil, err
	}
	closed := &closedFile{fh}
	s.files.Set(f.URI(), closed)
	return closed, nil
}

func (s *snapshot) IsOpen(uri span.URI) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.isOpenLocked(uri)

}

func (s *snapshot) openFiles() []source.VersionedFileHandle {
	s.mu.Lock()
	defer s.mu.Unlock()

	var open []source.VersionedFileHandle
	s.files.Range(func(uri span.URI, fh source.VersionedFileHandle) {
		if isFileOpen(fh) {
			open = append(open, fh)
		}
	})
	return open
}

func (s *snapshot) isOpenLocked(uri span.URI) bool {
	fh, _ := s.files.Get(uri)
	return isFileOpen(fh)
}

func isFileOpen(fh source.VersionedFileHandle) bool {
	_, open := fh.(*overlay)
	return open
}

func (s *snapshot) awaitLoaded(ctx context.Context) error {
	loadErr := s.awaitLoadedAllErrors(ctx)

	s.mu.Lock()
	defer s.mu.Unlock()

	// If we still have absolutely no metadata, check if the view failed to
	// initialize and return any errors.
	if s.useInvalidMetadata() && len(s.meta.metadata) > 0 {
		return nil
	}
	for _, m := range s.meta.metadata {
		if m.Valid {
			return nil
		}
	}
	if loadErr != nil {
		return loadErr.MainError
	}
	return nil
}

func (s *snapshot) GetCriticalError(ctx context.Context) *source.CriticalError {
	if wsErr := s.workspace.criticalError(ctx, s); wsErr != nil {
		return wsErr
	}

	loadErr := s.awaitLoadedAllErrors(ctx)
	if loadErr != nil && errors.Is(loadErr.MainError, context.Canceled) {
		return nil
	}

	// Even if packages didn't fail to load, we still may want to show
	// additional warnings.
	if loadErr == nil {
		wsPkgs, _ := s.ActivePackages(ctx)
		if msg := shouldShowAdHocPackagesWarning(s, wsPkgs); msg != "" {
			return &source.CriticalError{
				MainError: errors.New(msg),
			}
		}
		// Even if workspace packages were returned, there still may be an error
		// with the user's workspace layout. Workspace packages that only have the
		// ID "command-line-arguments" are usually a symptom of a bad workspace
		// configuration.
		//
		// TODO(rfindley): re-evaluate this heuristic.
		if containsCommandLineArguments(wsPkgs) {
			return s.workspaceLayoutError(ctx)
		}
		return nil
	}

	if errMsg := loadErr.MainError.Error(); strings.Contains(errMsg, "cannot find main module") || strings.Contains(errMsg, "go.mod file not found") {
		return s.workspaceLayoutError(ctx)
	}
	return loadErr
}

const adHocPackagesWarning = `You are outside of a module and outside of $GOPATH/src.
If you are using modules, please open your editor to a directory in your module.
If you believe this warning is incorrect, please file an issue: https://github.com/golang/go/issues/new.`

func shouldShowAdHocPackagesWarning(snapshot source.Snapshot, pkgs []source.Package) string {
	if snapshot.ValidBuildConfiguration() {
		return ""
	}
	for _, pkg := range pkgs {
		if len(pkg.MissingDependencies()) > 0 {
			return adHocPackagesWarning
		}
	}
	return ""
}

func containsCommandLineArguments(pkgs []source.Package) bool {
	for _, pkg := range pkgs {
		if source.IsCommandLineArguments(pkg.ID()) {
			return true
		}
	}
	return false
}

func (s *snapshot) awaitLoadedAllErrors(ctx context.Context) *source.CriticalError {
	// Do not return results until the snapshot's view has been initialized.
	s.AwaitInitialized(ctx)

	// TODO(rfindley): Should we be more careful about returning the
	// initialization error? Is it possible for the initialization error to be
	// corrected without a successful reinitialization?
	s.mu.Lock()
	initializedErr := s.initializedErr
	s.mu.Unlock()

	if initializedErr != nil {
		return initializedErr
	}

	// TODO(rfindley): revisit this handling. Calling reloadWorkspace with a
	// cancelled context should have the same effect, so this preemptive handling
	// should not be necessary.
	//
	// Also: GetCriticalError ignores context cancellation errors. Should we be
	// returning nil here?
	if ctx.Err() != nil {
		return &source.CriticalError{MainError: ctx.Err()}
	}

	// TODO(rfindley): reloading is not idempotent: if we try to reload or load
	// orphaned files below and fail, we won't try again. For that reason, we
	// could get different results from subsequent calls to this function, which
	// may cause critical errors to be suppressed.

	if err := s.reloadWorkspace(ctx); err != nil {
		diags := s.extractGoCommandErrors(ctx, err)
		return &source.CriticalError{
			MainError:   err,
			Diagnostics: diags,
		}
	}

	if err := s.reloadOrphanedFiles(ctx); err != nil {
		diags := s.extractGoCommandErrors(ctx, err)
		return &source.CriticalError{
			MainError:   err,
			Diagnostics: diags,
		}
	}
	return nil
}

func (s *snapshot) getInitializationError(ctx context.Context) *source.CriticalError {
	s.mu.Lock()
	defer s.mu.Unlock()

	return s.initializedErr
}

func (s *snapshot) AwaitInitialized(ctx context.Context) {
	select {
	case <-ctx.Done():
		return
	case <-s.view.initialWorkspaceLoad:
	}
	// We typically prefer to run something as intensive as the IWL without
	// blocking. I'm not sure if there is a way to do that here.
	s.initialize(ctx, false)
}

// reloadWorkspace reloads the metadata for all invalidated workspace packages.
func (s *snapshot) reloadWorkspace(ctx context.Context) error {
	var scopes []loadScope
	var seen map[PackagePath]bool
	s.mu.Lock()
	for _, pkgPaths := range s.shouldLoad {
		for _, pkgPath := range pkgPaths {
			if seen == nil {
				seen = make(map[PackagePath]bool)
			}
			if seen[pkgPath] {
				continue
			}
			seen[pkgPath] = true
			scopes = append(scopes, packageLoadScope(pkgPath))
		}
	}
	s.mu.Unlock()

	if len(scopes) == 0 {
		return nil
	}

	// If the view's build configuration is invalid, we cannot reload by
	// package path. Just reload the directory instead.
	if !s.ValidBuildConfiguration() {
		scopes = []loadScope{viewLoadScope("LOAD_INVALID_VIEW")}
	}

	err := s.load(ctx, false, scopes...)

	// Unless the context was canceled, set "shouldLoad" to false for all
	// of the metadata we attempted to load.
	if !errors.Is(err, context.Canceled) {
		s.clearShouldLoad(scopes...)
	}

	return err
}

func (s *snapshot) reloadOrphanedFiles(ctx context.Context) error {
	// When we load ./... or a package path directly, we may not get packages
	// that exist only in overlays. As a workaround, we search all of the files
	// available in the snapshot and reload their metadata individually using a
	// file= query if the metadata is unavailable.
	files := s.orphanedFiles()

	// Files without a valid package declaration can't be loaded. Don't try.
	var scopes []loadScope
	for _, file := range files {
		pgf, err := s.ParseGo(ctx, file, source.ParseHeader)
		if err != nil {
			continue
		}
		if !pgf.File.Package.IsValid() {
			continue
		}

		scopes = append(scopes, fileLoadScope(file.URI()))
	}

	if len(scopes) == 0 {
		return nil
	}

	// The regtests match this exact log message, keep them in sync.
	event.Log(ctx, "reloadOrphanedFiles reloading", tag.Query.Of(scopes))
	err := s.load(ctx, false, scopes...)

	// If we failed to load some files, i.e. they have no metadata,
	// mark the failures so we don't bother retrying until the file's
	// content changes.
	//
	// TODO(rstambler): This may be an overestimate if the load stopped
	// early for an unrelated errors. Add a fallback?
	//
	// Check for context cancellation so that we don't incorrectly mark files
	// as unloadable, but don't return before setting all workspace packages.
	if ctx.Err() == nil && err != nil {
		event.Error(ctx, "reloadOrphanedFiles: failed to load", err, tag.Query.Of(scopes))
		s.mu.Lock()
		for _, scope := range scopes {
			uri := span.URI(scope.(fileLoadScope))
			if s.noValidMetadataForURILocked(uri) {
				s.unloadableFiles[uri] = struct{}{}
			}
		}
		s.mu.Unlock()
	}
	return nil
}

func (s *snapshot) orphanedFiles() []source.VersionedFileHandle {
	s.mu.Lock()
	defer s.mu.Unlock()

	var files []source.VersionedFileHandle
	s.files.Range(func(uri span.URI, fh source.VersionedFileHandle) {
		// Don't try to reload metadata for go.mod files.
		if s.view.FileKind(fh) != source.Go {
			return
		}
		// If the URI doesn't belong to this view, then it's not in a workspace
		// package and should not be reloaded directly.
		if !source.InDir(s.view.folder.Filename(), uri.Filename()) {
			return
		}
		// If the file is not open and is in a vendor directory, don't treat it
		// like a workspace package.
		if _, ok := fh.(*overlay); !ok && inVendor(uri) {
			return
		}
		// Don't reload metadata for files we've already deemed unloadable.
		if _, ok := s.unloadableFiles[uri]; ok {
			return
		}
		if s.noValidMetadataForURILocked(uri) {
			files = append(files, fh)
		}
	})
	return files
}

// TODO(golang/go#53756): this function needs to consider more than just the
// absolute URI, for example:
//   - the position of /vendor/ with respect to the relevant module root
//   - whether or not go.work is in use (as vendoring isn't supported in workspace mode)
//
// Most likely, each call site of inVendor needs to be reconsidered to
// understand and correctly implement the desired behavior.
func inVendor(uri span.URI) bool {
	if !strings.Contains(string(uri), "/vendor/") {
		return false
	}
	// Only packages in _subdirectories_ of /vendor/ are considered vendored
	// (/vendor/a/foo.go is vendored, /vendor/foo.go is not).
	split := strings.Split(string(uri), "/vendor/")
	if len(split) < 2 {
		return false
	}
	return strings.Contains(split[1], "/")
}

// unappliedChanges is a file source that handles an uncloned snapshot.
type unappliedChanges struct {
	originalSnapshot *snapshot
	changes          map[span.URI]*fileChange
}

func (ac *unappliedChanges) GetFile(ctx context.Context, uri span.URI) (source.FileHandle, error) {
	if c, ok := ac.changes[uri]; ok {
		return c.fileHandle, nil
	}
	return ac.originalSnapshot.GetFile(ctx, uri)
}

func (s *snapshot) clone(ctx, bgCtx context.Context, changes map[span.URI]*fileChange, forceReloadMetadata bool) (*snapshot, func()) {
	ctx, done := event.Start(ctx, "snapshot.clone")
	defer done()

	newWorkspace, reinit := s.workspace.Clone(ctx, changes, &unappliedChanges{
		originalSnapshot: s,
		changes:          changes,
	})

	s.mu.Lock()
	defer s.mu.Unlock()

	// If there is an initialization error and a vendor directory changed, try to
	// reinit.
	if s.initializedErr != nil {
		for uri := range changes {
			if inVendor(uri) {
				reinit = true
				break
			}
		}
	}

	bgCtx, cancel := context.WithCancel(bgCtx)
	result := &snapshot{
		id:                   s.id + 1,
		store:                s.store,
		view:                 s.view,
		backgroundCtx:        bgCtx,
		cancel:               cancel,
		builtin:              s.builtin,
		initialized:          s.initialized,
		initializedErr:       s.initializedErr,
		packages:             s.packages.Clone(),
		isActivePackageCache: s.isActivePackageCache.Clone(),
		actions:              s.actions.Clone(),
		files:                s.files.Clone(),
		parsedGoFiles:        s.parsedGoFiles.Clone(),
		parseKeysByURI:       s.parseKeysByURI.Clone(),
		symbolizeHandles:     s.symbolizeHandles.Clone(),
		workspacePackages:    make(map[PackageID]PackagePath, len(s.workspacePackages)),
		unloadableFiles:      make(map[span.URI]struct{}, len(s.unloadableFiles)),
		parseModHandles:      s.parseModHandles.Clone(),
		parseWorkHandles:     s.parseWorkHandles.Clone(),
		modTidyHandles:       s.modTidyHandles.Clone(),
		modWhyHandles:        s.modWhyHandles.Clone(),
		knownSubdirs:         s.knownSubdirs.Clone(),
		workspace:            newWorkspace,
	}

	// The snapshot should be initialized if either s was uninitialized, or we've
	// detected a change that triggers reinitialization.
	if reinit {
		result.initialized = false
	}

	// Create a lease on the new snapshot.
	// (Best to do this early in case the code below hides an
	// incref/decref operation that might destroy it prematurely.)
	release := result.Acquire()

	// Copy the set of unloadable files.
	//
	// TODO(rfindley): this looks wrong. Shouldn't we clear unloadableFiles on
	// changes to environment or workspace layout, or more generally on any
	// metadata change?
	for k, v := range s.unloadableFiles {
		result.unloadableFiles[k] = v
	}

	// TODO(adonovan): merge loops over "changes".
	for uri := range changes {
		keys, ok := result.parseKeysByURI.Get(uri)
		if ok {
			for _, key := range keys {
				result.parsedGoFiles.Delete(key)
			}
			result.parseKeysByURI.Delete(uri)
		}

		// Invalidate go.mod-related handles.
		result.modTidyHandles.Delete(uri)
		result.modWhyHandles.Delete(uri)

		// Invalidate handles for cached symbols.
		result.symbolizeHandles.Delete(uri)
	}

	// Add all of the known subdirectories, but don't update them for the
	// changed files. We need to rebuild the workspace module to know the
	// true set of known subdirectories, but we don't want to do that in clone.
	result.knownSubdirs = s.knownSubdirs.Clone()
	result.knownSubdirsPatternCache = s.knownSubdirsPatternCache
	for _, c := range changes {
		result.unprocessedSubdirChanges = append(result.unprocessedSubdirChanges, c)
	}

	// directIDs keeps track of package IDs that have directly changed.
	// It maps id->invalidateMetadata.
	directIDs := map[PackageID]bool{}

	// Invalidate all package metadata if the workspace module has changed.
	if reinit {
		for k := range s.meta.metadata {
			directIDs[k] = true
		}
	}

	// Compute invalidations based on file changes.
	anyImportDeleted := false      // import deletions can resolve cycles
	anyFileOpenedOrClosed := false // opened files affect workspace packages
	anyFileAdded := false          // adding a file can resolve missing dependencies

	for uri, change := range changes {
		// The original FileHandle for this URI is cached on the snapshot.
		originalFH, _ := s.files.Get(uri)
		var originalOpen, newOpen bool
		_, originalOpen = originalFH.(*overlay)
		_, newOpen = change.fileHandle.(*overlay)
		anyFileOpenedOrClosed = anyFileOpenedOrClosed || (originalOpen != newOpen)
		anyFileAdded = anyFileAdded || (originalFH == nil && change.fileHandle != nil)

		// If uri is a Go file, check if it has changed in a way that would
		// invalidate metadata. Note that we can't use s.view.FileKind here,
		// because the file type that matters is not what the *client* tells us,
		// but what the Go command sees.
		var invalidateMetadata, pkgFileChanged, importDeleted bool
		if strings.HasSuffix(uri.Filename(), ".go") {
			invalidateMetadata, pkgFileChanged, importDeleted = metadataChanges(ctx, s, originalFH, change.fileHandle)
		}

		invalidateMetadata = invalidateMetadata || forceReloadMetadata || reinit
		anyImportDeleted = anyImportDeleted || importDeleted

		// Mark all of the package IDs containing the given file.
		filePackageIDs := invalidatedPackageIDs(uri, s.meta.ids, pkgFileChanged)
		for id := range filePackageIDs {
			directIDs[id] = directIDs[id] || invalidateMetadata
		}

		// Invalidate the previous modTidyHandle if any of the files have been
		// saved or if any of the metadata has been invalidated.
		if invalidateMetadata || fileWasSaved(originalFH, change.fileHandle) {
			// TODO(maybe): Only delete mod handles for
			// which the withoutURI is relevant.
			// Requires reverse-engineering the go command. (!)

			result.modTidyHandles.Clear()
			result.modWhyHandles.Clear()
		}

		result.parseModHandles.Delete(uri)
		result.parseWorkHandles.Delete(uri)
		// Handle the invalidated file; it may have new contents or not exist.
		if !change.exists {
			result.files.Delete(uri)
		} else {
			result.files.Set(uri, change.fileHandle)
		}

		// Make sure to remove the changed file from the unloadable set.
		delete(result.unloadableFiles, uri)
	}

	// Deleting an import can cause list errors due to import cycles to be
	// resolved. The best we can do without parsing the list error message is to
	// hope that list errors may have been resolved by a deleted import.
	//
	// We could do better by parsing the list error message. We already do this
	// to assign a better range to the list error, but for such critical
	// functionality as metadata, it's better to be conservative until it proves
	// impractical.
	//
	// We could also do better by looking at which imports were deleted and
	// trying to find cycles they are involved in. This fails when the file goes
	// from an unparseable state to a parseable state, as we don't have a
	// starting point to compare with.
	if anyImportDeleted {
		for id, metadata := range s.meta.metadata {
			if len(metadata.Errors) > 0 {
				directIDs[id] = true
			}
		}
	}

	// Adding a file can resolve missing dependencies from existing packages.
	//
	// We could be smart here and try to guess which packages may have been
	// fixed, but until that proves necessary, just invalidate metadata for any
	// package with missing dependencies.
	if anyFileAdded {
		for id, metadata := range s.meta.metadata {
			if len(metadata.MissingDeps) > 0 {
				directIDs[id] = true
			}
		}
	}

	// Invalidate reverse dependencies too.
	// idsToInvalidate keeps track of transitive reverse dependencies.
	// If an ID is present in the map, invalidate its types.
	// If an ID's value is true, invalidate its metadata too.
	idsToInvalidate := map[PackageID]bool{}
	var addRevDeps func(PackageID, bool)
	addRevDeps = func(id PackageID, invalidateMetadata bool) {
		current, seen := idsToInvalidate[id]
		newInvalidateMetadata := current || invalidateMetadata

		// If we've already seen this ID, and the value of invalidate
		// metadata has not changed, we can return early.
		if seen && current == newInvalidateMetadata {
			return
		}
		idsToInvalidate[id] = newInvalidateMetadata
		for _, rid := range s.meta.importedBy[id] {
			addRevDeps(rid, invalidateMetadata)
		}
	}
	for id, invalidateMetadata := range directIDs {
		addRevDeps(id, invalidateMetadata)
	}

	result.invalidatePackagesLocked(idsToInvalidate)

	// If a file has been deleted, we must delete metadata for all packages
	// containing that file.
	//
	// TODO(rfindley): why not keep invalid metadata in this case? If we
	// otherwise allow operate on invalid metadata, why not continue to do so,
	// skipping the missing file?
	skipID := map[PackageID]bool{}
	for _, c := range changes {
		if c.exists {
			continue
		}
		// The file has been deleted.
		if ids, ok := s.meta.ids[c.fileHandle.URI()]; ok {
			for _, id := range ids {
				skipID[id] = true
			}
		}
	}

	// Any packages that need loading in s still need loading in the new
	// snapshot.
	for k, v := range s.shouldLoad {
		if result.shouldLoad == nil {
			result.shouldLoad = make(map[PackageID][]PackagePath)
		}
		result.shouldLoad[k] = v
	}

	// TODO(rfindley): consolidate the this workspace mode detection with
	// workspace invalidation.
	workspaceModeChanged := s.workspaceMode() != result.workspaceMode()

	// We delete invalid metadata in the following cases:
	// - If we are forcing a reload of metadata.
	// - If the workspace mode has changed, as stale metadata may produce
	//   confusing or incorrect diagnostics.
	//
	// TODO(rfindley): we should probably also clear metadata if we are
	// reinitializing the workspace, as otherwise we could leave around a bunch
	// of irrelevant and duplicate metadata (for example, if the module path
	// changed). However, this breaks the "experimentalUseInvalidMetadata"
	// feature, which relies on stale metadata when, for example, a go.mod file
	// is broken via invalid syntax.
	deleteInvalidMetadata := forceReloadMetadata || workspaceModeChanged

	// Compute which metadata updates are required. We only need to invalidate
	// packages directly containing the affected file, and only if it changed in
	// a relevant way.
	metadataUpdates := make(map[PackageID]*KnownMetadata)
	for k, v := range s.meta.metadata {
		invalidateMetadata := idsToInvalidate[k]

		// For metadata that has been newly invalidated, capture package paths
		// requiring reloading in the shouldLoad map.
		if invalidateMetadata && !source.IsCommandLineArguments(string(v.ID)) {
			if result.shouldLoad == nil {
				result.shouldLoad = make(map[PackageID][]PackagePath)
			}
			needsReload := []PackagePath{v.PkgPath}
			if v.ForTest != "" && v.ForTest != v.PkgPath {
				// When reloading test variants, always reload their ForTest package as
				// well. Otherwise, we may miss test variants in the resulting load.
				//
				// TODO(rfindley): is this actually sufficient? Is it possible that
				// other test variants may be invalidated? Either way, we should
				// determine exactly what needs to be reloaded here.
				needsReload = append(needsReload, v.ForTest)
			}
			result.shouldLoad[k] = needsReload
		}

		// Check whether the metadata should be deleted.
		if skipID[k] || (invalidateMetadata && deleteInvalidMetadata) {
			metadataUpdates[k] = nil
			continue
		}

		// Check if the metadata has changed.
		valid := v.Valid && !invalidateMetadata
		if valid != v.Valid {
			// Mark invalidated metadata rather than deleting it outright.
			metadataUpdates[k] = &KnownMetadata{
				Metadata: v.Metadata,
				Valid:    valid,
			}
		}
	}

	// Update metadata, if necessary.
	result.meta = s.meta.Clone(metadataUpdates)

	// Update workspace and active packages, if necessary.
	if result.meta != s.meta || anyFileOpenedOrClosed {
		result.workspacePackages = computeWorkspacePackagesLocked(result, result.meta)
		result.resetIsActivePackageLocked()
	} else {
		result.workspacePackages = s.workspacePackages
	}

	// Don't bother copying the importedBy graph,
	// as it changes each time we update metadata.

	// If the snapshot's workspace mode has changed, the packages loaded using
	// the previous mode are no longer relevant, so clear them out.
	if workspaceModeChanged {
		result.workspacePackages = map[PackageID]PackagePath{}
	}
	result.dumpWorkspace("clone")
	return result, release
}

// invalidatedPackageIDs returns all packages invalidated by a change to uri.
// If we haven't seen this URI before, we guess based on files in the same
// directory. This is of course incorrect in build systems where packages are
// not organized by directory.
//
// If packageFileChanged is set, the file is either a new file, or has a new
// package name. In this case, all known packages in the directory will be
// invalidated.
func invalidatedPackageIDs(uri span.URI, known map[span.URI][]PackageID, packageFileChanged bool) map[PackageID]struct{} {
	invalidated := make(map[PackageID]struct{})

	// At a minimum, we invalidate packages known to contain uri.
	for _, id := range known[uri] {
		invalidated[id] = struct{}{}
	}

	// If the file didn't move to a new package, we should only invalidate the
	// packages it is currently contained inside.
	if !packageFileChanged && len(invalidated) > 0 {
		return invalidated
	}

	// This is a file we don't yet know about, or which has moved packages. Guess
	// relevant packages by considering files in the same directory.

	// Cache of FileInfo to avoid unnecessary stats for multiple files in the
	// same directory.
	stats := make(map[string]struct {
		os.FileInfo
		error
	})
	getInfo := func(dir string) (os.FileInfo, error) {
		if res, ok := stats[dir]; ok {
			return res.FileInfo, res.error
		}
		fi, err := os.Stat(dir)
		stats[dir] = struct {
			os.FileInfo
			error
		}{fi, err}
		return fi, err
	}
	dir := filepath.Dir(uri.Filename())
	fi, err := getInfo(dir)
	if err == nil {
		// Aggregate all possibly relevant package IDs.
		for knownURI, ids := range known {
			knownDir := filepath.Dir(knownURI.Filename())
			knownFI, err := getInfo(knownDir)
			if err != nil {
				continue
			}
			if os.SameFile(fi, knownFI) {
				for _, id := range ids {
					invalidated[id] = struct{}{}
				}
			}
		}
	}
	return invalidated
}

// invalidatePackagesLocked deletes data associated with the given package IDs.
//
// Note: all keys in the ids map are invalidated, regardless of the
// corresponding value.
//
// s.mu must be held while calling this function.
func (s *snapshot) invalidatePackagesLocked(ids map[PackageID]bool) {
	// Delete invalidated package type information.
	for id := range ids {
		for _, mode := range source.AllParseModes {
			key := packageKey{mode, id}
			s.packages.Delete(key)
		}
	}

	// Copy actions.
	// TODO(adonovan): opt: avoid iteration over s.actions.
	var actionsToDelete []actionKey
	s.actions.Range(func(k, _ interface{}) {
		key := k.(actionKey)
		if _, ok := ids[key.pkgid]; ok {
			actionsToDelete = append(actionsToDelete, key)
		}
	})
	for _, key := range actionsToDelete {
		s.actions.Delete(key)
	}
}

// fileWasSaved reports whether the FileHandle passed in has been saved. It
// accomplishes this by checking to see if the original and current FileHandles
// are both overlays, and if the current FileHandle is saved while the original
// FileHandle was not saved.
func fileWasSaved(originalFH, currentFH source.FileHandle) bool {
	c, ok := currentFH.(*overlay)
	if !ok || c == nil {
		return true
	}
	o, ok := originalFH.(*overlay)
	if !ok || o == nil {
		return c.saved
	}
	return !o.saved && c.saved
}

// metadataChanges detects features of the change from oldFH->newFH that may
// affect package metadata.
//
// It uses lockedSnapshot to access cached parse information. lockedSnapshot
// must be locked.
//
// The result parameters have the following meaning:
//   - invalidate means that package metadata for packages containing the file
//     should be invalidated.
//   - pkgFileChanged means that the file->package associates for the file have
//     changed (possibly because the file is new, or because its package name has
//     changed).
//   - importDeleted means that an import has been deleted, or we can't
//     determine if an import was deleted due to errors.
func metadataChanges(ctx context.Context, lockedSnapshot *snapshot, oldFH, newFH source.FileHandle) (invalidate, pkgFileChanged, importDeleted bool) {
	if oldFH == nil || newFH == nil { // existential changes
		changed := (oldFH == nil) != (newFH == nil)
		return changed, changed, (newFH == nil) // we don't know if an import was deleted
	}

	// If the file hasn't changed, there's no need to reload.
	if oldFH.FileIdentity() == newFH.FileIdentity() {
		return false, false, false
	}

	// Parse headers to compare package names and imports.
	oldHead, oldErr := peekOrParse(ctx, lockedSnapshot, oldFH, source.ParseHeader)
	newHead, newErr := peekOrParse(ctx, lockedSnapshot, newFH, source.ParseHeader)

	if oldErr != nil || newErr != nil {
		// TODO(rfindley): we can get here if newFH does not exists. There is
		// asymmetry here, in that newFH may be non-nil even if the underlying file
		// does not exist.
		//
		// We should not produce a non-nil filehandle for a file that does not exist.
		errChanged := (oldErr == nil) != (newErr == nil)
		return errChanged, errChanged, (newErr != nil) // we don't know if an import was deleted
	}

	// `go list` fails completely if the file header cannot be parsed. If we go
	// from a non-parsing state to a parsing state, we should reload.
	if oldHead.ParseErr != nil && newHead.ParseErr == nil {
		return true, true, true // We don't know what changed, so fall back on full invalidation.
	}

	// If a package name has changed, the set of package imports may have changed
	// in ways we can't detect here. Assume an import has been deleted.
	if oldHead.File.Name.Name != newHead.File.Name.Name {
		return true, true, true
	}

	// Check whether package imports have changed. Only consider potentially
	// valid imports paths.
	oldImports := validImports(oldHead.File.Imports)
	newImports := validImports(newHead.File.Imports)

	for path := range newImports {
		if _, ok := oldImports[path]; ok {
			delete(oldImports, path)
		} else {
			invalidate = true // a new, potentially valid import was added
		}
	}

	if len(oldImports) > 0 {
		invalidate = true
		importDeleted = true
	}

	// If the change does not otherwise invalidate metadata, get the full ASTs in
	// order to check magic comments.
	//
	// Note: if this affects performance we can probably avoid parsing in the
	// common case by first scanning the source for potential comments.
	if !invalidate {
		origFull, oldErr := peekOrParse(ctx, lockedSnapshot, oldFH, source.ParseFull)
		currFull, newErr := peekOrParse(ctx, lockedSnapshot, newFH, source.ParseFull)
		if oldErr == nil && newErr == nil {
			invalidate = magicCommentsChanged(origFull.File, currFull.File)
		} else {
			// At this point, we shouldn't ever fail to produce a ParsedGoFile, as
			// we're already past header parsing.
			bug.Reportf("metadataChanges: unparseable file %v (old error: %v, new error: %v)", oldFH.URI(), oldErr, newErr)
		}
	}

	return invalidate, pkgFileChanged, importDeleted
}

// peekOrParse returns the cached ParsedGoFile if it exists,
// otherwise parses without populating the cache.
//
// It returns an error if the file could not be read (note that parsing errors
// are stored in ParsedGoFile.ParseErr).
//
// lockedSnapshot must be locked.
func peekOrParse(ctx context.Context, lockedSnapshot *snapshot, fh source.FileHandle, mode source.ParseMode) (*source.ParsedGoFile, error) {
	// Peek in the cache without populating it.
	// We do this to reduce retained heap, not work.
	if parsed, _ := lockedSnapshot.peekParseGoLocked(fh, mode); parsed != nil {
		return parsed, nil // cache hit
	}
	return parseGoImpl(ctx, token.NewFileSet(), fh, mode)
}

func magicCommentsChanged(original *ast.File, current *ast.File) bool {
	oldComments := extractMagicComments(original)
	newComments := extractMagicComments(current)
	if len(oldComments) != len(newComments) {
		return true
	}
	for i := range oldComments {
		if oldComments[i] != newComments[i] {
			return true
		}
	}
	return false
}

// validImports extracts the set of valid import paths from imports.
func validImports(imports []*ast.ImportSpec) map[string]struct{} {
	m := make(map[string]struct{})
	for _, spec := range imports {
		if path := spec.Path.Value; validImportPath(path) {
			m[path] = struct{}{}
		}
	}
	return m
}

func validImportPath(path string) bool {
	path, err := strconv.Unquote(path)
	if err != nil {
		return false
	}
	if path == "" {
		return false
	}
	if path[len(path)-1] == '/' {
		return false
	}
	return true
}

var buildConstraintOrEmbedRe = regexp.MustCompile(`^//(go:embed|go:build|\s*\+build).*`)

// extractMagicComments finds magic comments that affect metadata in f.
func extractMagicComments(f *ast.File) []string {
	var results []string
	for _, cg := range f.Comments {
		for _, c := range cg.List {
			if buildConstraintOrEmbedRe.MatchString(c.Text) {
				results = append(results, c.Text)
			}
		}
	}
	return results
}

func (s *snapshot) BuiltinFile(ctx context.Context) (*source.ParsedGoFile, error) {
	s.AwaitInitialized(ctx)

	s.mu.Lock()
	builtin := s.builtin
	s.mu.Unlock()

	if builtin == "" {
		return nil, fmt.Errorf("no builtin package for view %s", s.view.name)
	}

	fh, err := s.GetFile(ctx, builtin)
	if err != nil {
		return nil, err
	}
	return s.ParseGo(ctx, fh, source.ParseFull)
}

func (s *snapshot) IsBuiltin(ctx context.Context, uri span.URI) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	// We should always get the builtin URI in a canonical form, so use simple
	// string comparison here. span.CompareURI is too expensive.
	return uri == s.builtin
}

func (s *snapshot) setBuiltin(path string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.builtin = span.URIFromPath(path)
}

// BuildGoplsMod generates a go.mod file for all modules in the workspace. It
// bypasses any existing gopls.mod.
func (s *snapshot) BuildGoplsMod(ctx context.Context) (*modfile.File, error) {
	allModules, err := findModules(s.view.folder, pathExcludedByFilterFunc(s.view.rootURI.Filename(), s.view.gomodcache, s.View().Options()), 0)
	if err != nil {
		return nil, err
	}
	return buildWorkspaceModFile(ctx, allModules, s)
}

// TODO(rfindley): move this to workspace.go
func buildWorkspaceModFile(ctx context.Context, modFiles map[span.URI]struct{}, fs source.FileSource) (*modfile.File, error) {
	file := &modfile.File{}
	file.AddModuleStmt("gopls-workspace")
	// Track the highest Go version, to be set on the workspace module.
	// Fall back to 1.12 -- old versions insist on having some version.
	goVersion := "1.12"

	paths := map[string]span.URI{}
	excludes := map[string][]string{}
	var sortedModURIs []span.URI
	for uri := range modFiles {
		sortedModURIs = append(sortedModURIs, uri)
	}
	sort.Slice(sortedModURIs, func(i, j int) bool {
		return sortedModURIs[i] < sortedModURIs[j]
	})
	for _, modURI := range sortedModURIs {
		fh, err := fs.GetFile(ctx, modURI)
		if err != nil {
			return nil, err
		}
		content, err := fh.Read()
		if err != nil {
			return nil, err
		}
		parsed, err := modfile.Parse(fh.URI().Filename(), content, nil)
		if err != nil {
			return nil, err
		}
		if file == nil || parsed.Module == nil {
			return nil, fmt.Errorf("no module declaration for %s", modURI)
		}
		// Prepend "v" to go versions to make them valid semver.
		if parsed.Go != nil && semver.Compare("v"+goVersion, "v"+parsed.Go.Version) < 0 {
			goVersion = parsed.Go.Version
		}
		path := parsed.Module.Mod.Path
		if seen, ok := paths[path]; ok {
			return nil, fmt.Errorf("found module %q multiple times in the workspace, at:\n\t%q\n\t%q", path, seen, modURI)
		}
		paths[path] = modURI
		// If the module's path includes a major version, we expect it to have
		// a matching major version.
		_, majorVersion, _ := module.SplitPathVersion(path)
		if majorVersion == "" {
			majorVersion = "/v0"
		}
		majorVersion = strings.TrimLeft(majorVersion, "/.") // handle gopkg.in versions
		file.AddNewRequire(path, source.WorkspaceModuleVersion(majorVersion), false)
		if err := file.AddReplace(path, "", dirURI(modURI).Filename(), ""); err != nil {
			return nil, err
		}
		for _, exclude := range parsed.Exclude {
			excludes[exclude.Mod.Path] = append(excludes[exclude.Mod.Path], exclude.Mod.Version)
		}
	}
	if goVersion != "" {
		file.AddGoStmt(goVersion)
	}
	// Go back through all of the modules to handle any of their replace
	// statements.
	for _, modURI := range sortedModURIs {
		fh, err := fs.GetFile(ctx, modURI)
		if err != nil {
			return nil, err
		}
		content, err := fh.Read()
		if err != nil {
			return nil, err
		}
		parsed, err := modfile.Parse(fh.URI().Filename(), content, nil)
		if err != nil {
			return nil, err
		}
		// If any of the workspace modules have replace directives, they need
		// to be reflected in the workspace module.
		for _, rep := range parsed.Replace {
			// Don't replace any modules that are in our workspace--we should
			// always use the version in the workspace.
			if _, ok := paths[rep.Old.Path]; ok {
				continue
			}
			newPath := rep.New.Path
			newVersion := rep.New.Version
			// If a replace points to a module in the workspace, make sure we
			// direct it to version of the module in the workspace.
			if m, ok := paths[rep.New.Path]; ok {
				newPath = dirURI(m).Filename()
				newVersion = ""
			} else if rep.New.Version == "" && !filepath.IsAbs(rep.New.Path) {
				// Make any relative paths absolute.
				newPath = filepath.Join(dirURI(modURI).Filename(), rep.New.Path)
			}
			if err := file.AddReplace(rep.Old.Path, rep.Old.Version, newPath, newVersion); err != nil {
				return nil, err
			}
		}
	}
	for path, versions := range excludes {
		for _, version := range versions {
			file.AddExclude(path, version)
		}
	}
	file.SortBlocks()
	return file, nil
}

func buildWorkspaceSumFile(ctx context.Context, modFiles map[span.URI]struct{}, fs source.FileSource) ([]byte, error) {
	allSums := map[module.Version][]string{}
	for modURI := range modFiles {
		// TODO(rfindley): factor out this pattern into a uripath package.
		sumURI := span.URIFromPath(filepath.Join(filepath.Dir(modURI.Filename()), "go.sum"))
		fh, err := fs.GetFile(ctx, sumURI)
		if err != nil {
			continue
		}
		data, err := fh.Read()
		if os.IsNotExist(err) {
			continue
		}
		if err != nil {
			return nil, fmt.Errorf("reading go sum: %w", err)
		}
		if err := readGoSum(allSums, sumURI.Filename(), data); err != nil {
			return nil, err
		}
	}
	// This logic to write go.sum is copied (with minor modifications) from
	// https://cs.opensource.google/go/go/+/master:src/cmd/go/internal/modfetch/fetch.go;l=631;drc=762eda346a9f4062feaa8a9fc0d17d72b11586f0
	var mods []module.Version
	for m := range allSums {
		mods = append(mods, m)
	}
	module.Sort(mods)

	var buf bytes.Buffer
	for _, m := range mods {
		list := allSums[m]
		sort.Strings(list)
		// Note (rfindley): here we add all sum lines without verification, because
		// the assumption is that if they come from a go.sum file, they are
		// trusted.
		for _, h := range list {
			fmt.Fprintf(&buf, "%s %s %s\n", m.Path, m.Version, h)
		}
	}
	return buf.Bytes(), nil
}

// readGoSum is copied (with minor modifications) from
// https://cs.opensource.google/go/go/+/master:src/cmd/go/internal/modfetch/fetch.go;l=398;drc=762eda346a9f4062feaa8a9fc0d17d72b11586f0
func readGoSum(dst map[module.Version][]string, file string, data []byte) error {
	lineno := 0
	for len(data) > 0 {
		var line []byte
		lineno++
		i := bytes.IndexByte(data, '\n')
		if i < 0 {
			line, data = data, nil
		} else {
			line, data = data[:i], data[i+1:]
		}
		f := strings.Fields(string(line))
		if len(f) == 0 {
			// blank line; skip it
			continue
		}
		if len(f) != 3 {
			return fmt.Errorf("malformed go.sum:\n%s:%d: wrong number of fields %v", file, lineno, len(f))
		}
		mod := module.Version{Path: f[0], Version: f[1]}
		dst[mod] = append(dst[mod], f[2])
	}
	return nil
}

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

	"golang.org/x/sync/errgroup"
	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/go/types/objectpath"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/gopls/internal/lsp/source/methodsets"
	"golang.org/x/tools/gopls/internal/lsp/source/xrefs"
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
	sequenceID uint64
	globalID   source.GlobalSnapshotID

	// TODO(rfindley): the snapshot holding a reference to the view poses
	// lifecycle problems: a view may be shut down and waiting for work
	// associated with this snapshot to complete. While most accesses of the view
	// are benign (options or workspace information), this is not formalized and
	// it is wrong for the snapshot to use a shutdown view.
	//
	// Fix this by passing options and workspace information to the snapshot,
	// both of which should be immutable for the snapshot.
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

	// parseCache holds an LRU cache of recently parsed files.
	parseCache *parseCache

	// symbolizeHandles maps each file URI to a handle for the future
	// result of computing the symbols declared in that file.
	symbolizeHandles *persistent.Map // from span.URI to *memoize.Promise[symbolizeResult]

	// packages maps a packageKey to a *packageHandle.
	// It may be invalidated when a file's content changes.
	//
	// Invariants to preserve:
	//  - packages.Get(id).meta == meta.metadata[id] for all ids
	//  - if a package is in packages, then all of its dependencies should also
	//    be in packages, unless there is a missing import
	packages *persistent.Map // from packageID to *packageHandle

	// activePackages maps a package ID to a memoized active package, or nil if
	// the package is known not to be open.
	//
	// IDs not contained in the map are not known to be open or not open.
	activePackages *persistent.Map // from packageID to *Package

	// analyses maps an analysisKey (which identifies a package
	// and a set of analyzers) to the handle for the future result
	// of loading the package and analyzing it.
	analyses *persistent.Map // from analysisKey to analysisPromise

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

	// TODO(rfindley): rename the handles below to "promises". A promise is
	// different from a handle (we mutate the package handle.)

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
	modVulnHandles *persistent.Map // from span.URI to *memoize.Promise[modVulnResult]

	// knownSubdirs is the set of subdirectories in the workspace, used to
	// create glob patterns for file watching.
	knownSubdirs             knownDirsSet
	knownSubdirsPatternCache string
	// unprocessedSubdirChanges are any changes that might affect the set of
	// subdirectories in the workspace. They are not reflected to knownSubdirs
	// during the snapshot cloning step as it can slow down cloning.
	unprocessedSubdirChanges []*fileChange

	// workspaceModFiles holds the set of mod files active in this snapshot.
	//
	// This is either empty, a single entry for the workspace go.mod file, or the
	// set of mod files used by the workspace go.work file.
	//
	// This set is immutable inside the snapshot, and therefore is not guarded by mu.
	workspaceModFiles    map[span.URI]struct{}
	workspaceModFilesErr error // error encountered computing workspaceModFiles
}

var globalSnapshotID uint64

func nextSnapshotID() source.GlobalSnapshotID {
	return source.GlobalSnapshotID(atomic.AddUint64(&globalSnapshotID, 1))
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
		log.Panicf("%d: acquire() after Destroy(%q)", s.globalID, *(*string)(destroyedBy))
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
		log.Panicf("%d: Destroy(%q) after Destroy(%q)", s.globalID, destroyedBy, *(*string)(old))
	}

	s.packages.Destroy()
	s.activePackages.Destroy()
	s.analyses.Destroy()
	s.files.Destroy()
	s.knownSubdirs.Destroy()
	s.symbolizeHandles.Destroy()
	s.parseModHandles.Destroy()
	s.parseWorkHandles.Destroy()
	s.modTidyHandles.Destroy()
	s.modVulnHandles.Destroy()
	s.modWhyHandles.Destroy()
}

func (s *snapshot) SequenceID() uint64 {
	return s.sequenceID
}

func (s *snapshot) GlobalID() source.GlobalSnapshotID {
	return s.globalID
}

func (s *snapshot) View() source.View {
	return s.view
}

func (s *snapshot) BackgroundContext() context.Context {
	return s.backgroundCtx
}

func (s *snapshot) ModFiles() []span.URI {
	var uris []span.URI
	for modURI := range s.workspaceModFiles {
		uris = append(uris, modURI)
	}
	return uris
}

func (s *snapshot) WorkFile() span.URI {
	return s.view.effectiveGOWORK()
}

func (s *snapshot) Templates() map[span.URI]source.FileHandle {
	s.mu.Lock()
	defer s.mu.Unlock()

	tmpls := map[span.URI]source.FileHandle{}
	s.files.Range(func(k span.URI, fh source.FileHandle) {
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
	if len(s.workspaceModFiles) > 0 {
		return true
	}
	// The user may have a multiple directories in their GOPATH.
	// Check if the workspace is within any of them.
	// TODO(rfindley): this should probably be subject to "if GO111MODULES = off {...}".
	for _, gp := range filepath.SplitList(s.view.gopath) {
		if source.InDir(filepath.Join(gp, "src"), s.view.folder.Filename()) {
			return true
		}
	}
	return false
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
func (s *snapshot) moduleMode() bool {
	// Since we only really understand the `go` command, if the user has a
	// different GOPACKAGESDRIVER, assume that their configuration is valid.
	if s.view.hasGopackagesDriver {
		return false
	}

	switch s.view.effectiveGO111MODULE() {
	case on:
		return true
	case off:
		return false
	default:
		return len(s.workspaceModFiles) > 0 || s.view.gowork != ""
	}
}

// workspaceMode describes the way in which the snapshot's workspace should
// be loaded.
//
// TODO(rfindley): remove this, in favor of specific methods.
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
	if len(s.workspaceModFiles) == 0 && validBuildConfiguration {
		return mode
	}
	mode |= moduleMode
	options := s.view.Options()
	if options.TempModfile {
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
			packages.NeedEmbedFiles |
			packages.LoadMode(packagesinternal.DepsErrors) |
			packages.LoadMode(packagesinternal.ForTest),
		Fset:    nil, // we do our own parsing
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
	packagesinternal.SetGoCmdRunner(cfg, s.view.gocmdRunner)
	return cfg
}

func (s *snapshot) RunGoCommandDirect(ctx context.Context, mode source.InvocationFlags, inv *gocommand.Invocation) (*bytes.Buffer, error) {
	_, inv, cleanup, err := s.goCommandInvocation(ctx, mode, inv)
	if err != nil {
		return nil, err
	}
	defer cleanup()

	return s.view.gocmdRunner.Run(ctx, *inv)
}

func (s *snapshot) RunGoCommandPiped(ctx context.Context, mode source.InvocationFlags, inv *gocommand.Invocation, stdout, stderr io.Writer) error {
	_, inv, cleanup, err := s.goCommandInvocation(ctx, mode, inv)
	if err != nil {
		return err
	}
	defer cleanup()
	return s.view.gocmdRunner.RunPiped(ctx, *inv, stdout, stderr)
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
		return s.view.gocmdRunner.Run(ctx, *inv)
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
	inv.Env = append(append(append(os.Environ(), s.view.options.EnvSlice()...), inv.Env...), "GO111MODULE="+s.view.GO111MODULE())
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
		if s.view.effectiveGOWORK() == "" && s.view.gomod != "" {
			modURI = s.view.gomod
		}
	} else {
		modURI = s.GoModForFile(span.URIFromPath(inv.WorkingDir))
	}

	var modContent []byte
	if modURI != "" {
		modFH, err := s.ReadFile(ctx, modURI)
		if err != nil {
			return "", nil, cleanup, err
		}
		modContent, err = modFH.Content()
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

	const mutableModFlag = "mod"
	// If the mod flag isn't set, populate it based on the mode and workspace.
	// TODO(rfindley): this doesn't make sense if we're not in module mode
	if inv.ModFlag == "" {
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
	if !needTempMod && s.view.gowork != "" {
		// Since we're running in the workspace root, the go command will resolve GOWORK automatically.
	} else if useTempMod {
		if modURI == "" {
			return "", nil, cleanup, fmt.Errorf("no go.mod file found in %s", inv.WorkingDir)
		}
		modFH, err := s.ReadFile(ctx, modURI)
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

func (s *snapshot) buildOverlay() map[string][]byte {
	s.mu.Lock()
	defer s.mu.Unlock()

	overlays := make(map[string][]byte)
	s.files.Range(func(uri span.URI, fh source.FileHandle) {
		overlay, ok := fh.(*Overlay)
		if !ok {
			return
		}
		if overlay.saved {
			return
		}
		// TODO(rstambler): Make sure not to send overlays outside of the current view.
		overlays[uri.Filename()] = overlay.content
	})
	return overlays
}

// Package data kinds, identifying various package data that may be stored in
// the file cache.
const (
	xrefsKind       = "xrefs"
	methodSetsKind  = "methodsets"
	exportDataKind  = "export"
	diagnosticsKind = "diagnostics"
)

func (s *snapshot) PackageDiagnostics(ctx context.Context, ids ...PackageID) (map[span.URI][]*source.Diagnostic, error) {
	// TODO(rfindley): opt: avoid unnecessary encode->decode after type-checking.
	data, err := s.getPackageData(ctx, diagnosticsKind, ids, func(p *syntaxPackage) []byte {
		return encodeDiagnostics(p.diagnostics)
	})
	perFile := make(map[span.URI][]*source.Diagnostic)
	for _, data := range data {
		if data != nil {
			for _, diag := range data.m.Diagnostics {
				perFile[diag.URI] = append(perFile[diag.URI], diag)
			}
			diags := decodeDiagnostics(data.data)
			for _, diag := range diags {
				perFile[diag.URI] = append(perFile[diag.URI], diag)
			}
		}
	}
	return perFile, err
}

func (s *snapshot) References(ctx context.Context, ids ...PackageID) ([]source.XrefIndex, error) {
	data, err := s.getPackageData(ctx, xrefsKind, ids, func(p *syntaxPackage) []byte { return p.xrefs })
	indexes := make([]source.XrefIndex, len(ids))
	for i, data := range data {
		if data != nil {
			indexes[i] = XrefIndex{m: data.m, data: data.data}
		}
	}
	return indexes, err
}

// An XrefIndex is a helper for looking up a package in a given package.
type XrefIndex struct {
	m    *source.Metadata
	data []byte
}

func (index XrefIndex) Lookup(targets map[PackagePath]map[objectpath.Path]struct{}) []protocol.Location {
	return xrefs.Lookup(index.m, index.data, targets)
}

func (s *snapshot) MethodSets(ctx context.Context, ids ...PackageID) ([]*methodsets.Index, error) {
	// TODO(rfindley): opt: avoid unnecessary encode->decode after type-checking.
	data, err := s.getPackageData(ctx, methodSetsKind, ids, func(p *syntaxPackage) []byte {
		return p.methodsets.Encode()
	})
	indexes := make([]*methodsets.Index, len(ids))
	for i, data := range data {
		if data != nil {
			indexes[i] = methodsets.Decode(data.data)
		} else if ids[i] == "unsafe" {
			indexes[i] = &methodsets.Index{}
		} else {
			panic(fmt.Sprintf("nil data for %s", ids[i]))
		}
	}
	return indexes, err
}

func (s *snapshot) MetadataForFile(ctx context.Context, uri span.URI) ([]*source.Metadata, error) {
	s.mu.Lock()

	// Start with the set of package associations derived from the last load.
	ids := s.meta.ids[uri]

	shouldLoad := false // whether any packages containing uri are marked 'shouldLoad'
	for _, id := range ids {
		if len(s.shouldLoad[id]) > 0 {
			shouldLoad = true
		}
	}

	// Check if uri is known to be unloadable.
	_, unloadable := s.unloadableFiles[uri]

	s.mu.Unlock()

	// Reload if loading is likely to improve the package associations for uri:
	//  - uri is not contained in any valid packages
	//  - ...or one of the packages containing uri is marked 'shouldLoad'
	//  - ...but uri is not unloadable
	if (shouldLoad || len(ids) == 0) && !unloadable {
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
		// Furthermore, the result of MetadataForFile should be consistent upon
		// subsequent calls, even if the file is marked as unloadable.
		if err != nil && !errors.Is(err, errNoPackages) {
			event.Error(ctx, "MetadataForFile", err)
		}
	}

	// Retrieve the metadata.
	s.mu.Lock()
	defer s.mu.Unlock()
	ids = s.meta.ids[uri]
	metas := make([]*source.Metadata, len(ids))
	for i, id := range ids {
		metas[i] = s.meta.metadata[id]
		if metas[i] == nil {
			panic("nil metadata")
		}
	}
	// Metadata is only ever added by loading,
	// so if we get here and still have
	// no IDs, uri is unloadable.
	if !unloadable && len(ids) == 0 {
		s.unloadableFiles[uri] = struct{}{}
	}

	// Sort packages "narrowest" to "widest" (in practice: non-tests before tests).
	sort.Slice(metas, func(i, j int) bool {
		return len(metas[i].CompiledGoFiles) < len(metas[j].CompiledGoFiles)
	})

	return metas, nil
}

func (s *snapshot) ReverseDependencies(ctx context.Context, id PackageID, transitive bool) (map[PackageID]*source.Metadata, error) {
	if err := s.awaitLoaded(ctx); err != nil {
		return nil, err
	}
	s.mu.Lock()
	meta := s.meta
	s.mu.Unlock()

	var rdeps map[PackageID]*source.Metadata
	if transitive {
		rdeps = meta.reverseReflexiveTransitiveClosure(id)

		// Remove the original package ID from the map.
		// (Callers all want irreflexivity but it's easier
		// to compute reflexively then subtract.)
		delete(rdeps, id)

	} else {
		// direct reverse dependencies
		rdeps = make(map[PackageID]*source.Metadata)
		for _, rdepID := range meta.importedBy[id] {
			if rdep := meta.metadata[rdepID]; rdep != nil {
				rdeps[rdepID] = rdep
			}
		}
	}

	return rdeps, nil
}

func (s *snapshot) workspaceMetadata() (meta []*source.Metadata) {
	s.mu.Lock()
	defer s.mu.Unlock()

	for id := range s.workspacePackages {
		meta = append(meta, s.meta.metadata[id])
	}
	return meta
}

// -- Active package tracking --
//
// We say a package is "active" if any of its files are open. After
// type-checking we keep active packages in memory. The activePackages
// peristent map does bookkeeping for the set of active packages.

// getActivePackage returns a the memoized active package for id, if it exists.
// If id is not active or has not yet been type-checked, it returns nil.
func (s *snapshot) getActivePackage(id PackageID) *Package {
	s.mu.Lock()
	defer s.mu.Unlock()

	if value, ok := s.activePackages.Get(id); ok {
		return value.(*Package) // possibly nil, if we have already checked this id.
	}
	return nil
}

// memoizeActivePackage checks if pkg is active, and if so either records it in
// the active packages map or returns the existing memoized active package for id.
//
// The resulting package is non-nil if and only if the specified package is open.
func (s *snapshot) memoizeActivePackage(id PackageID, pkg *Package) (active *Package) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if value, ok := s.activePackages.Get(id); ok {
		return value.(*Package) // possibly nil, if we have already checked this id.
	}

	defer func() {
		s.activePackages.Set(id, active, nil) // store the result either way: remember that pkg is not open
	}()
	for _, cgf := range pkg.Metadata().GoFiles {
		if s.isOpenLocked(cgf) {
			return pkg
		}
	}
	for _, cgf := range pkg.Metadata().CompiledGoFiles {
		if s.isOpenLocked(cgf) {
			return pkg
		}
	}
	return nil
}

func (s *snapshot) resetActivePackagesLocked() {
	s.activePackages.Destroy()
	s.activePackages = persistent.NewMap(packageIDLessInterface)
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

	// If GOWORK is outside the folder, ensure we are watching it.
	gowork := s.view.effectiveGOWORK()
	if gowork != "" && !source.InDir(s.view.folder.Filename(), gowork.Filename()) {
		patterns[gowork.Filename()] = struct{}{}
	}

	// Add a pattern for each Go module in the workspace that is not within the view.
	dirs := s.dirs(ctx)
	for _, dir := range dirs {
		dirName := dir.Filename()

		// If the directory is within the view's folder, we're already watching
		// it with the first pattern above.
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
	dirs := s.dirs(ctx)

	s.mu.Lock()
	defer s.mu.Unlock()

	s.knownSubdirs.Destroy()
	s.knownSubdirs = newKnownDirsSet()
	s.knownSubdirsPatternCache = ""
	s.files.Range(func(uri span.URI, fh source.FileHandle) {
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

	s.files.Range(func(uri span.URI, fh source.FileHandle) {
		if source.InDir(dir.Filename(), uri.Filename()) {
			files = append(files, uri)
		}
	})
	return files
}

func (s *snapshot) ActiveMetadata(ctx context.Context) ([]*source.Metadata, error) {
	if err := s.awaitLoaded(ctx); err != nil {
		return nil, err
	}
	return s.workspaceMetadata(), nil
}

// Symbols extracts and returns symbol information for every file contained in
// a loaded package. It awaits snapshot loading.
//
// TODO(rfindley): move this to the top of cache/symbols.go
func (s *snapshot) Symbols(ctx context.Context) (map[span.URI][]source.Symbol, error) {
	if err := s.awaitLoaded(ctx); err != nil {
		return nil, err
	}

	// Build symbols for all loaded Go files.
	s.mu.Lock()
	meta := s.meta
	s.mu.Unlock()

	goFiles := make(map[span.URI]struct{})
	for _, m := range meta.metadata {
		for _, uri := range m.GoFiles {
			goFiles[uri] = struct{}{}
		}
		for _, uri := range m.CompiledGoFiles {
			goFiles[uri] = struct{}{}
		}
	}

	// Symbolize them in parallel.
	var (
		group    errgroup.Group
		nprocs   = 2 * runtime.GOMAXPROCS(-1) // symbolize is a mix of I/O and CPU
		resultMu sync.Mutex
		result   = make(map[span.URI][]source.Symbol)
	)
	group.SetLimit(nprocs)
	for uri := range goFiles {
		uri := uri
		group.Go(func() error {
			symbols, err := s.symbolize(ctx, uri)
			if err != nil {
				return err
			}
			resultMu.Lock()
			result[uri] = symbols
			resultMu.Unlock()
			return nil
		})
	}
	// Keep going on errors, but log the first failure.
	// Partial results are better than no symbol results.
	if err := group.Wait(); err != nil {
		event.Error(ctx, "getting snapshot symbols", err)
	}
	return result, nil
}

func (s *snapshot) AllMetadata(ctx context.Context) ([]*source.Metadata, error) {
	if err := s.awaitLoaded(ctx); err != nil {
		return nil, err
	}

	s.mu.Lock()
	g := s.meta
	s.mu.Unlock()

	meta := make([]*source.Metadata, 0, len(g.metadata))
	for _, m := range g.metadata {
		meta = append(meta, m)
	}
	return meta, nil
}

// TODO(rfindley): clarify that this is only active modules. Or update to just
// use findRootPattern.
func (s *snapshot) GoModForFile(uri span.URI) span.URI {
	return moduleForURI(s.workspaceModFiles, uri)
}

func moduleForURI(modFiles map[span.URI]struct{}, uri span.URI) span.URI {
	var match span.URI
	for modURI := range modFiles {
		if !source.InDir(span.Dir(modURI).Filename(), uri.Filename()) {
			continue
		}
		if len(modURI) > len(match) {
			match = modURI
		}
	}
	return match
}

// nearestModFile finds the nearest go.mod file contained in the directory
// containing uri, or a parent of that directory.
//
// The given uri must be a file, not a directory.
func nearestModFile(ctx context.Context, uri span.URI, fs source.FileSource) (span.URI, error) {
	// TODO(rfindley)
	dir := filepath.Dir(uri.Filename())
	mod, err := findRootPattern(ctx, dir, "go.mod", fs)
	if err != nil {
		return "", err
	}
	return span.URIFromPath(mod), nil
}

func (s *snapshot) Metadata(id PackageID) *source.Metadata {
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
	for _, id := range s.meta.ids[uri] {
		if _, ok := s.meta.metadata[id]; ok {
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

func (s *snapshot) FindFile(uri span.URI) source.FileHandle {
	s.view.markKnown(uri)

	s.mu.Lock()
	defer s.mu.Unlock()

	result, _ := s.files.Get(uri)
	return result
}

// ReadFile returns a File for the given URI. If the file is unknown it is added
// to the managed set.
//
// ReadFile succeeds even if the file does not exist. A non-nil error return
// indicates some type of internal error, for example if ctx is cancelled.
func (s *snapshot) ReadFile(ctx context.Context, uri span.URI) (source.FileHandle, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	return lockedSnapshot{s}.ReadFile(ctx, uri)
}

// A lockedSnapshot implements the source.FileSource interface while holding
// the lock for the wrapped snapshot.
type lockedSnapshot struct{ wrapped *snapshot }

func (s lockedSnapshot) ReadFile(ctx context.Context, uri span.URI) (source.FileHandle, error) {
	s.wrapped.view.markKnown(uri)
	if fh, ok := s.wrapped.files.Get(uri); ok {
		return fh, nil
	}

	fh, err := s.wrapped.view.fs.ReadFile(ctx, uri)
	if err != nil {
		return nil, err
	}
	s.wrapped.files.Set(uri, fh)
	return fh, nil
}

func (s *snapshot) IsOpen(uri span.URI) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.isOpenLocked(uri)

}

func (s *snapshot) openFiles() []source.FileHandle {
	s.mu.Lock()
	defer s.mu.Unlock()

	var open []source.FileHandle
	s.files.Range(func(uri span.URI, fh source.FileHandle) {
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

func isFileOpen(fh source.FileHandle) bool {
	_, open := fh.(*Overlay)
	return open
}

func (s *snapshot) awaitLoaded(ctx context.Context) error {
	loadErr := s.awaitLoadedAllErrors(ctx)

	// TODO(rfindley): eliminate this function as part of simplifying
	// CriticalErrors.
	if loadErr != nil {
		return loadErr.MainError
	}
	return nil
}

func (s *snapshot) GetCriticalError(ctx context.Context) *source.CriticalError {
	// If we couldn't compute workspace mod files, then the load below is
	// invalid.
	//
	// TODO(rfindley): is this a clear error to present to the user?
	if s.workspaceModFilesErr != nil {
		return &source.CriticalError{MainError: s.workspaceModFilesErr}
	}

	loadErr := s.awaitLoadedAllErrors(ctx)
	if loadErr != nil && errors.Is(loadErr.MainError, context.Canceled) {
		return nil
	}

	// Even if packages didn't fail to load, we still may want to show
	// additional warnings.
	if loadErr == nil {
		active, _ := s.ActiveMetadata(ctx)
		if msg := shouldShowAdHocPackagesWarning(s, active); msg != "" {
			return &source.CriticalError{
				MainError: errors.New(msg),
			}
		}
		// Even if workspace packages were returned, there still may be an error
		// with the user's workspace layout. Workspace packages that only have the
		// ID "command-line-arguments" are usually a symptom of a bad workspace
		// configuration.
		//
		// This heuristic is path-dependent: we only get command-line-arguments
		// packages when we've loaded using file scopes, which only occurs
		// on-demand or via orphaned file reloading.
		//
		// TODO(rfindley): re-evaluate this heuristic.
		if containsCommandLineArguments(active) {
			err, diags := s.workspaceLayoutError(ctx)
			if err != nil {
				if ctx.Err() != nil {
					return nil // see the API documentation for source.Snapshot
				}
				return &source.CriticalError{
					MainError:   err,
					Diagnostics: diags,
				}
			}
		}
		return nil
	}

	if errMsg := loadErr.MainError.Error(); strings.Contains(errMsg, "cannot find main module") || strings.Contains(errMsg, "go.mod file not found") {
		err, diags := s.workspaceLayoutError(ctx)
		if err != nil {
			if ctx.Err() != nil {
				return nil // see the API documentation for source.Snapshot
			}
			return &source.CriticalError{
				MainError:   err,
				Diagnostics: diags,
			}
		}
	}
	return loadErr
}

// A portion of this text is expected by TestBrokenWorkspace_OutsideModule.
const adHocPackagesWarning = `You are outside of a module and outside of $GOPATH/src.
If you are using modules, please open your editor to a directory in your module.
If you believe this warning is incorrect, please file an issue: https://github.com/golang/go/issues/new.`

func shouldShowAdHocPackagesWarning(snapshot source.Snapshot, active []*source.Metadata) string {
	if !snapshot.ValidBuildConfiguration() {
		for _, m := range active {
			// A blank entry in DepsByImpPath
			// indicates a missing dependency.
			for _, importID := range m.DepsByImpPath {
				if importID == "" {
					return adHocPackagesWarning
				}
			}
		}
	}
	return ""
}

func containsCommandLineArguments(metas []*source.Metadata) bool {
	for _, m := range metas {
		if source.IsCommandLineArguments(m.ID) {
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
	if err := s.getInitializationError(); err != nil {
		return err
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

	if err := s.reloadOrphanedOpenFiles(ctx); err != nil {
		diags := s.extractGoCommandErrors(ctx, err)
		return &source.CriticalError{
			MainError:   err,
			Diagnostics: diags,
		}
	}
	return nil
}

func (s *snapshot) getInitializationError() *source.CriticalError {
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

func (s *snapshot) reloadOrphanedOpenFiles(ctx context.Context) error {
	// When we load ./... or a package path directly, we may not get packages
	// that exist only in overlays. As a workaround, we search all of the files
	// available in the snapshot and reload their metadata individually using a
	// file= query if the metadata is unavailable.
	files := s.orphanedOpenFiles()

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

func (s *snapshot) orphanedOpenFiles() []source.FileHandle {
	s.mu.Lock()
	defer s.mu.Unlock()

	var files []source.FileHandle
	s.files.Range(func(uri span.URI, fh source.FileHandle) {
		// Only consider open files, which will be represented as overlays.
		if _, isOverlay := fh.(*Overlay); !isOverlay {
			return
		}
		// Don't try to reload metadata for go.mod files.
		if s.view.FileKind(fh) != source.Go {
			return
		}
		// If the URI doesn't belong to this view, then it's not in a workspace
		// package and should not be reloaded directly.
		if !source.InDir(s.view.folder.Filename(), uri.Filename()) {
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
	_, after, found := cut(string(uri), "/vendor/")
	// Only subdirectories of /vendor/ are considered vendored
	// (/vendor/a/foo.go is vendored, /vendor/foo.go is not).
	return found && strings.Contains(after, "/")
}

// TODO(adonovan): replace with strings.Cut when we can assume go1.18.
func cut(s, sep string) (before, after string, found bool) {
	if i := strings.Index(s, sep); i >= 0 {
		return s[:i], s[i+len(sep):], true
	}
	return s, "", false
}

// unappliedChanges is a file source that handles an uncloned snapshot.
type unappliedChanges struct {
	originalSnapshot *snapshot
	changes          map[span.URI]*fileChange
}

func (ac *unappliedChanges) ReadFile(ctx context.Context, uri span.URI) (source.FileHandle, error) {
	if c, ok := ac.changes[uri]; ok {
		return c.fileHandle, nil
	}
	return ac.originalSnapshot.ReadFile(ctx, uri)
}

func (s *snapshot) clone(ctx, bgCtx context.Context, changes map[span.URI]*fileChange, forceReloadMetadata bool) (*snapshot, func()) {
	ctx, done := event.Start(ctx, "snapshot.clone")
	defer done()

	reinit := false
	wsModFiles, wsModFilesErr := s.workspaceModFiles, s.workspaceModFilesErr

	if workURI := s.view.effectiveGOWORK(); workURI != "" {
		if change, ok := changes[workURI]; ok {
			wsModFiles, wsModFilesErr = computeWorkspaceModFiles(ctx, s.view.gomod, workURI, s.view.effectiveGO111MODULE(), &unappliedChanges{
				originalSnapshot: s,
				changes:          changes,
			})
			// TODO(rfindley): don't rely on 'isUnchanged' here. Use a content hash instead.
			reinit = change.fileHandle.Saved() && !change.isUnchanged
		}
	}

	// Reinitialize if any workspace mod file has changed on disk.
	for uri, change := range changes {
		if _, ok := wsModFiles[uri]; ok && change.fileHandle.Saved() && !change.isUnchanged {
			reinit = true
		}
	}

	// Finally, process sumfile changes that may affect loading.
	for uri, change := range changes {
		if !change.fileHandle.Saved() {
			continue // like with go.mod files, we only reinit when things are saved
		}
		if filepath.Base(uri.Filename()) == "go.work.sum" && s.view.gowork != "" {
			if filepath.Dir(uri.Filename()) == filepath.Dir(s.view.gowork) {
				reinit = true
			}
		}
		if filepath.Base(uri.Filename()) == "go.sum" {
			dir := filepath.Dir(uri.Filename())
			modURI := span.URIFromPath(filepath.Join(dir, "go.mod"))
			if _, active := wsModFiles[modURI]; active {
				reinit = true
			}
		}
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	// Changes to vendor tree may require reinitialization,
	// either because of an initialization error
	// (e.g. "inconsistent vendoring detected"), or because
	// one or more modules may have moved into or out of the
	// vendor tree after 'go mod vendor' or 'rm -fr vendor/'.
	for uri := range changes {
		if inVendor(uri) && s.initializedErr != nil ||
			strings.HasSuffix(string(uri), "/vendor/modules.txt") {
			reinit = true
			break
		}
	}

	bgCtx, cancel := context.WithCancel(bgCtx)
	result := &snapshot{
		sequenceID:           s.sequenceID + 1,
		globalID:             nextSnapshotID(),
		store:                s.store,
		view:                 s.view,
		backgroundCtx:        bgCtx,
		cancel:               cancel,
		builtin:              s.builtin,
		initialized:          s.initialized,
		initializedErr:       s.initializedErr,
		packages:             s.packages.Clone(),
		activePackages:       s.activePackages.Clone(),
		analyses:             s.analyses.Clone(),
		files:                s.files.Clone(),
		parseCache:           s.parseCache,
		symbolizeHandles:     s.symbolizeHandles.Clone(),
		workspacePackages:    make(map[PackageID]PackagePath, len(s.workspacePackages)),
		unloadableFiles:      make(map[span.URI]struct{}, len(s.unloadableFiles)),
		parseModHandles:      s.parseModHandles.Clone(),
		parseWorkHandles:     s.parseWorkHandles.Clone(),
		modTidyHandles:       s.modTidyHandles.Clone(),
		modWhyHandles:        s.modWhyHandles.Clone(),
		modVulnHandles:       s.modVulnHandles.Clone(),
		knownSubdirs:         s.knownSubdirs.Clone(),
		workspaceModFiles:    wsModFiles,
		workspaceModFilesErr: wsModFilesErr,
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
	//
	// Maybe not, as major configuration changes cause a new view.
	for k, v := range s.unloadableFiles {
		result.unloadableFiles[k] = v
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
	// Note: this is not a set, it's a map from id to invalidateMetadata.
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
		// Invalidate go.mod-related handles.
		result.modTidyHandles.Delete(uri)
		result.modWhyHandles.Delete(uri)
		result.modVulnHandles.Delete(uri)

		// Invalidate handles for cached symbols.
		result.symbolizeHandles.Delete(uri)

		// The original FileHandle for this URI is cached on the snapshot.
		originalFH, _ := s.files.Get(uri)
		var originalOpen, newOpen bool
		_, originalOpen = originalFH.(*Overlay)
		_, newOpen = change.fileHandle.(*Overlay)
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
			directIDs[id] = directIDs[id] || invalidateMetadata // may insert 'false'
		}

		// Invalidate the previous modTidyHandle if any of the files have been
		// saved or if any of the metadata has been invalidated.
		if invalidateMetadata || fileWasSaved(originalFH, change.fileHandle) {
			// TODO(maybe): Only delete mod handles for
			// which the withoutURI is relevant.
			// Requires reverse-engineering the go command. (!)
			result.modTidyHandles.Clear()
			result.modWhyHandles.Clear()
			result.modVulnHandles.Clear()
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
			for _, impID := range metadata.DepsByImpPath {
				if impID == "" { // missing import
					directIDs[id] = true
					break
				}
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

	// Delete invalidated package type information.
	for id := range idsToInvalidate {
		result.packages.Delete(id)
		result.activePackages.Delete(id)
	}

	// Delete invalidated analysis actions.
	var actionsToDelete []analysisKey
	result.analyses.Range(func(k, _ interface{}) {
		key := k.(analysisKey)
		if _, ok := idsToInvalidate[key.pkgid]; ok {
			actionsToDelete = append(actionsToDelete, key)
		}
	})
	for _, key := range actionsToDelete {
		result.analyses.Delete(key)
	}

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

	// Compute which metadata updates are required. We only need to invalidate
	// packages directly containing the affected file, and only if it changed in
	// a relevant way.
	metadataUpdates := make(map[PackageID]*source.Metadata)
	for k, v := range s.meta.metadata {
		invalidateMetadata := idsToInvalidate[k]

		// For metadata that has been newly invalidated, capture package paths
		// requiring reloading in the shouldLoad map.
		if invalidateMetadata && !source.IsCommandLineArguments(v.ID) {
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
		if skipID[k] || invalidateMetadata {
			metadataUpdates[k] = nil
			continue
		}
	}

	// Update metadata, if necessary.
	result.meta = s.meta.Clone(metadataUpdates)

	// Update workspace and active packages, if necessary.
	if result.meta != s.meta || anyFileOpenedOrClosed {
		result.workspacePackages = computeWorkspacePackagesLocked(result, result.meta)
		result.resetActivePackagesLocked()
	} else {
		result.workspacePackages = s.workspacePackages
	}

	// Don't bother copying the importedBy graph,
	// as it changes each time we update metadata.

	// TODO(rfindley): consolidate the this workspace mode detection with
	// workspace invalidation.
	workspaceModeChanged := s.workspaceMode() != result.workspaceMode()

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

// fileWasSaved reports whether the FileHandle passed in has been saved. It
// accomplishes this by checking to see if the original and current FileHandles
// are both overlays, and if the current FileHandle is saved while the original
// FileHandle was not saved.
func fileWasSaved(originalFH, currentFH source.FileHandle) bool {
	c, ok := currentFH.(*Overlay)
	if !ok || c == nil {
		return true
	}
	o, ok := originalFH.(*Overlay)
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

	fset := token.NewFileSet()
	// Parse headers to compare package names and imports.
	oldHeads, oldErr := lockedSnapshot.parseCache.parseFiles(ctx, fset, source.ParseHeader, oldFH)
	newHeads, newErr := lockedSnapshot.parseCache.parseFiles(ctx, fset, source.ParseHeader, newFH)

	if oldErr != nil || newErr != nil {
		// TODO(rfindley): we can get here if newFH does not exist. There is
		// asymmetry, in that newFH may be non-nil even if the underlying file does
		// not exist.
		//
		// We should not produce a non-nil filehandle for a file that does not exist.
		errChanged := (oldErr == nil) != (newErr == nil)
		return errChanged, errChanged, (newErr != nil) // we don't know if an import was deleted
	}

	oldHead := oldHeads[0]
	newHead := newHeads[0]

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
		origFulls, oldErr := lockedSnapshot.parseCache.parseFiles(ctx, fset, source.ParseFull, oldFH)
		newFulls, newErr := lockedSnapshot.parseCache.parseFiles(ctx, fset, source.ParseFull, newFH)
		if oldErr == nil && newErr == nil {
			invalidate = magicCommentsChanged(origFulls[0].File, newFulls[0].File)
		} else {
			// At this point, we shouldn't ever fail to produce a ParsedGoFile, as
			// we're already past header parsing.
			bug.Reportf("metadataChanges: unparseable file %v (old error: %v, new error: %v)", oldFH.URI(), oldErr, newErr)
		}
	}

	return invalidate, pkgFileChanged, importDeleted
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

	fh, err := s.ReadFile(ctx, builtin)
	if err != nil {
		return nil, err
	}
	// For the builtin file only, we need syntactic object resolution
	// (since we can't type check).
	mode := source.ParseFull &^ source.SkipObjectResolution
	return parseGoImpl(ctx, token.NewFileSet(), fh, mode)
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

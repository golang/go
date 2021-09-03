// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"bytes"
	"context"
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"

	"golang.org/x/mod/modfile"
	"golang.org/x/mod/module"
	"golang.org/x/mod/semver"
	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/gocommand"
	"golang.org/x/tools/internal/lsp/debug/log"
	"golang.org/x/tools/internal/lsp/debug/tag"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/memoize"
	"golang.org/x/tools/internal/packagesinternal"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/typesinternal"
	errors "golang.org/x/xerrors"
)

type snapshot struct {
	memoize.Arg // allow as a memoize.Function arg

	id   uint64
	view *View

	cancel        func()
	backgroundCtx context.Context

	// the cache generation that contains the data for this snapshot.
	generation *memoize.Generation

	// The snapshot's initialization state is controlled by the fields below.
	//
	// initializeOnce guards snapshot initialization. Each snapshot is
	// initialized at most once: reinitialization is triggered on later snapshots
	// by invalidating this field.
	initializeOnce *sync.Once
	// initializedErr holds the last error resulting from initialization. If
	// initialization fails, we only retry when the the workspace modules change,
	// to avoid too many go/packages calls.
	initializedErr *source.CriticalError

	// mu guards all of the maps in the snapshot, as well as the builtin URI.
	mu sync.Mutex

	// builtin pins the AST and package for builtin.go in memory.
	builtin span.URI

	// ids maps file URIs to package IDs.
	// It may be invalidated on calls to go/packages.
	ids map[span.URI][]PackageID

	// metadata maps file IDs to their associated metadata.
	// It may invalidated on calls to go/packages.
	metadata map[PackageID]*KnownMetadata

	// importedBy maps package IDs to the list of packages that import them.
	importedBy map[PackageID][]PackageID

	// files maps file URIs to their corresponding FileHandles.
	// It may invalidated when a file's content changes.
	files map[span.URI]source.VersionedFileHandle

	// goFiles maps a parseKey to its parseGoHandle.
	goFiles map[parseKey]*parseGoHandle

	// TODO(rfindley): consider merging this with files to reduce burden on clone.
	symbols map[span.URI]*symbolHandle

	// packages maps a packageKey to a set of packageHandles to which that file belongs.
	// It may be invalidated when a file's content changes.
	packages map[packageKey]*packageHandle

	// actions maps an actionkey to its actionHandle.
	actions map[actionKey]*actionHandle

	// workspacePackages contains the workspace's packages, which are loaded
	// when the view is created.
	workspacePackages map[PackageID]PackagePath

	// unloadableFiles keeps track of files that we've failed to load.
	unloadableFiles map[span.URI]struct{}

	// parseModHandles keeps track of any ParseModHandles for the snapshot.
	// The handles need not refer to only the view's go.mod file.
	parseModHandles map[span.URI]*parseModHandle

	// Preserve go.mod-related handles to avoid garbage-collecting the results
	// of various calls to the go command. The handles need not refer to only
	// the view's go.mod file.
	modTidyHandles map[span.URI]*modTidyHandle
	modWhyHandles  map[span.URI]*modWhyHandle

	workspace          *workspace
	workspaceDirHandle *memoize.Handle

	// knownSubdirs is the set of subdirectories in the workspace, used to
	// create glob patterns for file watching.
	knownSubdirs map[span.URI]struct{}
	// unprocessedSubdirChanges are any changes that might affect the set of
	// subdirectories in the workspace. They are not reflected to knownSubdirs
	// during the snapshot cloning step as it can slow down cloning.
	unprocessedSubdirChanges []*fileChange
}

type packageKey struct {
	mode source.ParseMode
	id   PackageID
}

type actionKey struct {
	pkg      packageKey
	analyzer *analysis.Analyzer
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

func (s *snapshot) Templates() map[span.URI]source.VersionedFileHandle {
	if !s.view.options.ExperimentalTemplateSupport {
		return nil
	}
	ans := map[span.URI]source.VersionedFileHandle{}
	for k, x := range s.files {
		if strings.HasSuffix(filepath.Ext(k.Filename()), "tmpl") {
			ans[k] = x
		}
	}
	return ans
}

func (s *snapshot) ValidBuildConfiguration() bool {
	return validBuildConfiguration(s.view.rootURI, &s.view.workspaceInformation, s.workspace.getActiveModFiles())
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
	// If the user is intentionally limiting their workspace scope, don't
	// enable multi-module workspace mode.
	// TODO(rstambler): This should only change the calculation of the root,
	// not the mode.
	if !options.ExpandWorkspaceToModule {
		return mode
	}
	// The workspace module has been disabled by the user.
	if !options.ExperimentalWorkspaceModule {
		return mode
	}
	mode |= usesWorkspaceModule
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
			packages.NeedModule,
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

func (s *snapshot) goCommandInvocation(ctx context.Context, flags source.InvocationFlags, inv *gocommand.Invocation) (tmpURI span.URI, updatedInv *gocommand.Invocation, cleanup func(), err error) {
	s.view.optionsMu.Lock()
	allowModfileModificationOption := s.view.options.AllowModfileModifications
	allowNetworkOption := s.view.options.AllowImplicitNetworkAccess
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

	var modURI span.URI
	// Select the module context to use.
	// If we're type checking, we need to use the workspace context, meaning
	// the main (workspace) module. Otherwise, we should use the module for
	// the passed-in working dir.
	if mode == source.LoadWorkspace {
		if s.workspaceMode()&usesWorkspaceModule == 0 {
			for m := range s.workspace.getActiveModFiles() { // range to access the only element
				modURI = m
			}
		} else {
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

	vendorEnabled, err := s.vendorEnabled(ctx, modURI, modContent)
	if err != nil {
		return "", nil, cleanup, err
	}

	// If the mod flag isn't set, populate it based on the mode and workspace.
	if inv.ModFlag == "" {
		mutableModFlag := ""
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
		case source.UpdateUserModFile, source.WriteTemporaryModFile:
			inv.ModFlag = mutableModFlag
		}
	}

	wantTempMod := mode != source.UpdateUserModFile
	needTempMod := mode == source.WriteTemporaryModFile
	tempMod := wantTempMod && s.workspaceMode()&tempModfile != 0
	if needTempMod && !tempMod {
		return "", nil, cleanup, source.ErrTmpModfileUnsupported
	}

	if tempMod {
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

func (s *snapshot) buildOverlay() map[string][]byte {
	s.mu.Lock()
	defer s.mu.Unlock()

	overlays := make(map[string][]byte)
	for uri, fh := range s.files {
		overlay, ok := fh.(*overlay)
		if !ok {
			continue
		}
		if overlay.saved {
			continue
		}
		// TODO(rstambler): Make sure not to send overlays outside of the current view.
		overlays[uri.Filename()] = overlay.text
	}
	return overlays
}

func hashUnsavedOverlays(files map[span.URI]source.VersionedFileHandle) string {
	var unsaved []string
	for uri, fh := range files {
		if overlay, ok := fh.(*overlay); ok && !overlay.saved {
			unsaved = append(unsaved, uri.Filename())
		}
	}
	sort.Strings(unsaved)
	return hashContents([]byte(strings.Join(unsaved, "")))
}

func (s *snapshot) PackagesForFile(ctx context.Context, uri span.URI, mode source.TypecheckMode, includeTestVariants bool) ([]source.Package, error) {
	ctx = event.Label(ctx, tag.URI.Of(uri))

	phs, err := s.packageHandlesForFile(ctx, uri, mode, includeTestVariants)
	if err != nil {
		return nil, err
	}
	var pkgs []source.Package
	for _, ph := range phs {
		pkg, err := ph.check(ctx, s)
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
		return nil, errors.Errorf("no packages")
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
		return nil, errors.Errorf("no packages in input")
	}

	return ph.check(ctx, s)
}

func (s *snapshot) packageHandlesForFile(ctx context.Context, uri span.URI, mode source.TypecheckMode, includeTestVariants bool) ([]*packageHandle, error) {
	// Check if we should reload metadata for the file. We don't invalidate IDs
	// (though we should), so the IDs will be a better source of truth than the
	// metadata. If there are no IDs for the file, then we should also reload.
	fh, err := s.GetFile(ctx, uri)
	if err != nil {
		return nil, err
	}
	if fh.Kind() != source.Go {
		return nil, fmt.Errorf("no packages for non-Go file %s", uri)
	}
	knownIDs, err := s.getOrLoadIDsForURI(ctx, uri)
	if err != nil {
		return nil, err
	}

	var phs []*packageHandle
	for _, id := range knownIDs {
		// Filter out any intermediate test variants. We typically aren't
		// interested in these packages for file= style queries.
		if m := s.getMetadata(id); m != nil && m.IsIntermediateTestVariant && !includeTestVariants {
			continue
		}
		var parseModes []source.ParseMode
		switch mode {
		case source.TypecheckAll:
			if s.workspaceParseMode(id) == source.ParseFull {
				parseModes = []source.ParseMode{source.ParseFull}
			} else {
				parseModes = []source.ParseMode{source.ParseExported, source.ParseFull}
			}
		case source.TypecheckFull:
			parseModes = []source.ParseMode{source.ParseFull}
		case source.TypecheckWorkspace:
			parseModes = []source.ParseMode{s.workspaceParseMode(id)}
		}

		for _, parseMode := range parseModes {
			ph, err := s.buildPackageHandle(ctx, id, parseMode)
			if err != nil {
				return nil, err
			}
			phs = append(phs, ph)
		}
	}
	return phs, nil
}

func (s *snapshot) getOrLoadIDsForURI(ctx context.Context, uri span.URI) ([]PackageID, error) {
	knownIDs := s.getIDsForURI(uri)
	reload := len(knownIDs) == 0
	for _, id := range knownIDs {
		// Reload package metadata if any of the metadata has missing
		// dependencies, in case something has changed since the last time we
		// reloaded it.
		if s.noValidMetadataForID(id) {
			reload = true
			break
		}
		// TODO(golang/go#36918): Previously, we would reload any package with
		// missing dependencies. This is expensive and results in too many
		// calls to packages.Load. Determine what we should do instead.
	}
	if reload {
		err := s.load(ctx, false, fileURI(uri))

		if !s.useInvalidMetadata() && err != nil {
			return nil, err
		}
		// We've tried to reload and there are still no known IDs for the URI.
		// Return the load error, if there was one.
		knownIDs = s.getIDsForURI(uri)
		if len(knownIDs) == 0 {
			return nil, err
		}
	}
	return knownIDs, nil
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
	ids := make(map[PackageID]struct{})
	s.transitiveReverseDependencies(PackageID(id), ids)

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
	return ph.check(ctx, s)
}

// transitiveReverseDependencies populates the ids map with package IDs
// belonging to the provided package and its transitive reverse dependencies.
func (s *snapshot) transitiveReverseDependencies(id PackageID, ids map[PackageID]struct{}) {
	if _, ok := ids[id]; ok {
		return
	}
	m := s.getMetadata(id)
	// Only use invalid metadata if we support it.
	if m == nil || !(m.Valid || s.useInvalidMetadata()) {
		return
	}
	ids[id] = struct{}{}
	importedBy := s.getImportedBy(id)
	for _, parentID := range importedBy {
		s.transitiveReverseDependencies(parentID, ids)
	}
}

func (s *snapshot) getGoFile(key parseKey) *parseGoHandle {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.goFiles[key]
}

func (s *snapshot) addGoFile(key parseKey, pgh *parseGoHandle) *parseGoHandle {
	s.mu.Lock()
	defer s.mu.Unlock()
	if existing, ok := s.goFiles[key]; ok {
		return existing
	}
	s.goFiles[key] = pgh
	return pgh
}

func (s *snapshot) getParseModHandle(uri span.URI) *parseModHandle {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.parseModHandles[uri]
}

func (s *snapshot) getModWhyHandle(uri span.URI) *modWhyHandle {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.modWhyHandles[uri]
}

func (s *snapshot) getModTidyHandle(uri span.URI) *modTidyHandle {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.modTidyHandles[uri]
}

func (s *snapshot) getImportedBy(id PackageID) []PackageID {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.getImportedByLocked(id)
}

func (s *snapshot) getImportedByLocked(id PackageID) []PackageID {
	// If we haven't rebuilt the import graph since creating the snapshot.
	if len(s.importedBy) == 0 {
		s.rebuildImportGraph()
	}
	return s.importedBy[id]
}

func (s *snapshot) clearAndRebuildImportGraph() {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Completely invalidate the original map.
	s.importedBy = make(map[PackageID][]PackageID)
	s.rebuildImportGraph()
}

func (s *snapshot) rebuildImportGraph() {
	for id, m := range s.metadata {
		for _, importID := range m.Deps {
			s.importedBy[importID] = append(s.importedBy[importID], id)
		}
	}
}

func (s *snapshot) addPackageHandle(ph *packageHandle) *packageHandle {
	s.mu.Lock()
	defer s.mu.Unlock()

	// If the package handle has already been cached,
	// return the cached handle instead of overriding it.
	if ph, ok := s.packages[ph.packageKey()]; ok {
		return ph
	}
	s.packages[ph.packageKey()] = ph
	return ph
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

	seen := make(map[PackageID]bool)
	for id := range s.workspacePackages {
		if s.isActiveLocked(id, seen) {
			ids = append(ids, id)
		}
	}
	return ids
}

func (s *snapshot) isActiveLocked(id PackageID, seen map[PackageID]bool) (active bool) {
	if seen == nil {
		seen = make(map[PackageID]bool)
	}
	if seen, ok := seen[id]; ok {
		return seen
	}
	defer func() {
		seen[id] = active
	}()
	m, ok := s.metadata[id]
	if !ok {
		return false
	}
	for _, cgf := range m.CompiledGoFiles {
		if s.isOpenLocked(cgf) {
			return true
		}
	}
	for _, dep := range m.Deps {
		if s.isActiveLocked(dep, seen) {
			return true
		}
	}
	return false
}

func (s *snapshot) getWorkspacePkgPath(id PackageID) PackagePath {
	s.mu.Lock()
	defer s.mu.Unlock()

	return s.workspacePackages[id]
}

const fileExtensions = "go,mod,sum,work,tmpl"

func (s *snapshot) fileWatchingGlobPatterns(ctx context.Context) map[string]struct{} {
	// Work-around microsoft/vscode#100870 by making sure that we are,
	// at least, watching the user's entire workspace. This will still be
	// applied to every folder in the workspace.
	patterns := map[string]struct{}{
		fmt.Sprintf("**/*.{%s}", fileExtensions): {},
		"**/*.*tmpl":                             {},
	}
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
		patterns[fmt.Sprintf("%s/**/*.{%s}", dirName, fileExtensions)] = struct{}{}
	}

	// Some clients do not send notifications for changes to directories that
	// contain Go code (golang/go#42348). To handle this, explicitly watch all
	// of the directories in the workspace. We find them by adding the
	// directories of every file in the snapshot's workspace directories.
	var dirNames []string
	for _, uri := range s.getKnownSubdirs(dirs) {
		dirNames = append(dirNames, uri.Filename())
	}
	sort.Strings(dirNames)
	if len(dirNames) > 0 {
		patterns[fmt.Sprintf("{%s}", strings.Join(dirNames, ","))] = struct{}{}
	}
	return patterns
}

// collectAllKnownSubdirs collects all of the subdirectories within the
// snapshot's workspace directories. None of the workspace directories are
// included.
func (s *snapshot) collectAllKnownSubdirs(ctx context.Context) {
	dirs := s.workspace.dirs(ctx, s)

	s.mu.Lock()
	defer s.mu.Unlock()

	s.knownSubdirs = map[span.URI]struct{}{}
	for uri := range s.files {
		s.addKnownSubdirLocked(uri, dirs)
	}
}

func (s *snapshot) getKnownSubdirs(wsDirs []span.URI) []span.URI {
	s.mu.Lock()
	defer s.mu.Unlock()

	// First, process any pending changes and update the set of known
	// subdirectories.
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

	var result []span.URI
	for uri := range s.knownSubdirs {
		result = append(result, uri)
	}
	return result
}

func (s *snapshot) addKnownSubdirLocked(uri span.URI, dirs []span.URI) {
	dir := filepath.Dir(uri.Filename())
	// First check if the directory is already known, because then we can
	// return early.
	if _, ok := s.knownSubdirs[span.URIFromPath(dir)]; ok {
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
		if _, ok := s.knownSubdirs[uri]; ok {
			break
		}
		s.knownSubdirs[uri] = struct{}{}
		dir = filepath.Dir(dir)
	}
}

func (s *snapshot) removeKnownSubdirLocked(uri span.URI) {
	dir := filepath.Dir(uri.Filename())
	for dir != "" {
		uri := span.URIFromPath(dir)
		if _, ok := s.knownSubdirs[uri]; !ok {
			break
		}
		if info, _ := os.Stat(dir); info == nil {
			delete(s.knownSubdirs, uri)
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

	for uri := range s.files {
		if source.InDir(dir.Filename(), uri.Filename()) {
			files = append(files, uri)
		}
	}
	return files
}

func (s *snapshot) workspacePackageHandles(ctx context.Context) ([]*packageHandle, error) {
	if err := s.awaitLoaded(ctx); err != nil {
		return nil, err
	}
	var phs []*packageHandle
	for _, pkgID := range s.workspacePackageIDs() {
		ph, err := s.buildPackageHandle(ctx, pkgID, s.workspaceParseMode(pkgID))
		if err != nil {
			return nil, err
		}
		phs = append(phs, ph)
	}
	return phs, nil
}

func (s *snapshot) ActivePackages(ctx context.Context) ([]source.Package, error) {
	phs, err := s.activePackageHandles(ctx)
	if err != nil {
		return nil, err
	}
	var pkgs []source.Package
	for _, ph := range phs {
		pkg, err := ph.check(ctx, s)
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

func (s *snapshot) Symbols(ctx context.Context) (map[span.URI][]source.Symbol, error) {
	result := make(map[span.URI][]source.Symbol)
	for uri, f := range s.files {
		sh := s.buildSymbolHandle(ctx, f)
		v, err := sh.handle.Get(ctx, s.generation, s)
		if err != nil {
			return nil, err
		}
		data := v.(*symbolData)
		result[uri] = data.symbols
	}
	return result, nil
}

func (s *snapshot) MetadataForFile(ctx context.Context, uri span.URI) ([]source.Metadata, error) {
	knownIDs, err := s.getOrLoadIDsForURI(ctx, uri)
	if err != nil {
		return nil, err
	}
	var mds []source.Metadata
	for _, id := range knownIDs {
		md := s.getMetadata(id)
		mds = append(mds, md)
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
	for id := range s.metadata {
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

func (s *snapshot) CachedImportPaths(ctx context.Context) (map[string]source.Package, error) {
	// Don't reload workspace package metadata.
	// This function is meant to only return currently cached information.
	s.AwaitInitialized(ctx)

	s.mu.Lock()
	defer s.mu.Unlock()

	results := map[string]source.Package{}
	for _, ph := range s.packages {
		cachedPkg, err := ph.cached(s.generation)
		if err != nil {
			continue
		}
		for importPath, newPkg := range cachedPkg.imports {
			if oldPkg, ok := results[string(importPath)]; ok {
				// Using the same trick as NarrowestPackage, prefer non-variants.
				if len(newPkg.compiledGoFiles) < len(oldPkg.(*pkg).compiledGoFiles) {
					results[string(importPath)] = newPkg
				}
			} else {
				results[string(importPath)] = newPkg
			}
		}
	}
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

func (s *snapshot) getPackage(id PackageID, mode source.ParseMode) *packageHandle {
	s.mu.Lock()
	defer s.mu.Unlock()

	key := packageKey{
		id:   id,
		mode: mode,
	}
	return s.packages[key]
}

func (s *snapshot) getSymbolHandle(uri span.URI) *symbolHandle {
	s.mu.Lock()
	defer s.mu.Unlock()

	return s.symbols[uri]
}

func (s *snapshot) addSymbolHandle(sh *symbolHandle) *symbolHandle {
	s.mu.Lock()
	defer s.mu.Unlock()

	uri := sh.fh.URI()
	// If the package handle has already been cached,
	// return the cached handle instead of overriding it.
	if sh, ok := s.symbols[uri]; ok {
		return sh
	}
	s.symbols[uri] = sh
	return sh
}

func (s *snapshot) getActionHandle(id PackageID, m source.ParseMode, a *analysis.Analyzer) *actionHandle {
	s.mu.Lock()
	defer s.mu.Unlock()

	key := actionKey{
		pkg: packageKey{
			id:   id,
			mode: m,
		},
		analyzer: a,
	}
	return s.actions[key]
}

func (s *snapshot) addActionHandle(ah *actionHandle) *actionHandle {
	s.mu.Lock()
	defer s.mu.Unlock()

	key := actionKey{
		analyzer: ah.analyzer,
		pkg: packageKey{
			id:   ah.pkg.m.ID,
			mode: ah.pkg.mode,
		},
	}
	if ah, ok := s.actions[key]; ok {
		return ah
	}
	s.actions[key] = ah
	return ah
}

func (s *snapshot) getIDsForURI(uri span.URI) []PackageID {
	s.mu.Lock()
	defer s.mu.Unlock()

	return s.ids[uri]
}

func (s *snapshot) getMetadata(id PackageID) *KnownMetadata {
	s.mu.Lock()
	defer s.mu.Unlock()

	return s.metadata[id]
}

func (s *snapshot) shouldLoad(scope interface{}) bool {
	s.mu.Lock()
	defer s.mu.Unlock()

	switch scope := scope.(type) {
	case PackagePath:
		var meta *KnownMetadata
		for _, m := range s.metadata {
			if m.PkgPath != scope {
				continue
			}
			meta = m
		}
		if meta == nil || meta.ShouldLoad {
			return true
		}
		return false
	case fileURI:
		uri := span.URI(scope)
		ids := s.ids[uri]
		if len(ids) == 0 {
			return true
		}
		for _, id := range ids {
			m, ok := s.metadata[id]
			if !ok || m.ShouldLoad {
				return true
			}
		}
		return false
	default:
		return true
	}
}

func (s *snapshot) clearShouldLoad(scope interface{}) {
	s.mu.Lock()
	defer s.mu.Unlock()

	switch scope := scope.(type) {
	case PackagePath:
		var meta *KnownMetadata
		for _, m := range s.metadata {
			if m.PkgPath == scope {
				meta = m
			}
		}
		if meta == nil {
			return
		}
		meta.ShouldLoad = false
	case fileURI:
		uri := span.URI(scope)
		ids := s.ids[uri]
		if len(ids) == 0 {
			return
		}
		for _, id := range ids {
			if m, ok := s.metadata[id]; ok {
				m.ShouldLoad = false
			}
		}
	}
}

// noValidMetadataForURILocked reports whether there is any valid metadata for
// the given URI.
func (s *snapshot) noValidMetadataForURILocked(uri span.URI) bool {
	ids, ok := s.ids[uri]
	if !ok {
		return true
	}
	for _, id := range ids {
		if m, ok := s.metadata[id]; ok && m.Valid {
			return false
		}
	}
	return true
}

// noValidMetadataForID reports whether there is no valid metadata for the
// given ID.
func (s *snapshot) noValidMetadataForID(id PackageID) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.noValidMetadataForIDLocked(id)
}

func (s *snapshot) noValidMetadataForIDLocked(id PackageID) bool {
	m := s.metadata[id]
	return m == nil || !m.Valid
}

// updateIDForURIsLocked adds the given ID to the set of known IDs for the given URI.
// Any existing invalid IDs are removed from the set of known IDs. IDs that are
// not "command-line-arguments" are preferred, so if a new ID comes in for a
// URI that previously only had "command-line-arguments", the new ID will
// replace the "command-line-arguments" ID.
func (s *snapshot) updateIDForURIsLocked(id PackageID, uris map[span.URI]struct{}) {
	for uri := range uris {
		// Collect the new set of IDs, preserving any valid existing IDs.
		newIDs := []PackageID{id}
		for _, existingID := range s.ids[uri] {
			// Don't set duplicates of the same ID.
			if existingID == id {
				continue
			}
			// If the package previously only had a command-line-arguments ID,
			// delete the command-line-arguments workspace package.
			if source.IsCommandLineArguments(string(existingID)) {
				delete(s.workspacePackages, existingID)
				continue
			}
			// If the metadata for an existing ID is invalid, and we are
			// setting metadata for a new, valid ID--don't preserve the old ID.
			if m, ok := s.metadata[existingID]; !ok || !m.Valid {
				continue
			}
			newIDs = append(newIDs, existingID)
		}
		sort.Slice(newIDs, func(i, j int) bool {
			return newIDs[i] < newIDs[j]
		})
		s.ids[uri] = newIDs
	}
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

	return s.files[f.URI()]
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
	if fh, ok := s.files[f.URI()]; ok {
		return fh, nil
	}

	fh, err := s.view.session.cache.getFile(ctx, f.URI())
	if err != nil {
		return nil, err
	}
	closed := &closedFile{fh}
	s.files[f.URI()] = closed
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
	for _, fh := range s.files {
		if s.isOpenLocked(fh.URI()) {
			open = append(open, fh)
		}
	}
	return open
}

func (s *snapshot) isOpenLocked(uri span.URI) bool {
	_, open := s.files[uri].(*overlay)
	return open
}

func (s *snapshot) awaitLoaded(ctx context.Context) error {
	loadErr := s.awaitLoadedAllErrors(ctx)

	s.mu.Lock()
	defer s.mu.Unlock()

	// If we still have absolutely no metadata, check if the view failed to
	// initialize and return any errors.
	if s.useInvalidMetadata() && len(s.metadata) > 0 {
		return nil
	}
	for _, m := range s.metadata {
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

	// TODO(rstambler): Should we be more careful about returning the
	// initialization error? Is it possible for the initialization error to be
	// corrected without a successful reinitialization?
	s.mu.Lock()
	initializedErr := s.initializedErr
	s.mu.Unlock()
	if initializedErr != nil {
		return initializedErr
	}

	if ctx.Err() != nil {
		return &source.CriticalError{MainError: ctx.Err()}
	}

	if err := s.reloadWorkspace(ctx); err != nil {
		diags, _ := s.extractGoCommandErrors(ctx, err.Error())
		return &source.CriticalError{
			MainError: err,
			DiagList:  diags,
		}
	}
	if err := s.reloadOrphanedFiles(ctx); err != nil {
		diags, _ := s.extractGoCommandErrors(ctx, err.Error())
		return &source.CriticalError{
			MainError: err,
			DiagList:  diags,
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
	// See which of the workspace packages are missing metadata.
	s.mu.Lock()
	missingMetadata := len(s.workspacePackages) == 0 || len(s.metadata) == 0
	pkgPathSet := map[PackagePath]struct{}{}
	for id, pkgPath := range s.workspacePackages {
		if m, ok := s.metadata[id]; ok && m.Valid {
			continue
		}
		missingMetadata = true

		// Don't try to reload "command-line-arguments" directly.
		if source.IsCommandLineArguments(string(pkgPath)) {
			continue
		}
		pkgPathSet[pkgPath] = struct{}{}
	}
	s.mu.Unlock()

	// If the view's build configuration is invalid, we cannot reload by
	// package path. Just reload the directory instead.
	if missingMetadata && !s.ValidBuildConfiguration() {
		return s.load(ctx, false, viewLoadScope("LOAD_INVALID_VIEW"))
	}

	if len(pkgPathSet) == 0 {
		return nil
	}

	var pkgPaths []interface{}
	for pkgPath := range pkgPathSet {
		pkgPaths = append(pkgPaths, pkgPath)
	}
	return s.load(ctx, false, pkgPaths...)
}

func (s *snapshot) reloadOrphanedFiles(ctx context.Context) error {
	// When we load ./... or a package path directly, we may not get packages
	// that exist only in overlays. As a workaround, we search all of the files
	// available in the snapshot and reload their metadata individually using a
	// file= query if the metadata is unavailable.
	files := s.orphanedFiles()

	// Files without a valid package declaration can't be loaded. Don't try.
	var scopes []interface{}
	for _, file := range files {
		pgf, err := s.ParseGo(ctx, file, source.ParseHeader)
		if err != nil {
			continue
		}
		if !pgf.File.Package.IsValid() {
			continue
		}
		scopes = append(scopes, fileURI(file.URI()))
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
			uri := span.URI(scope.(fileURI))
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
	for uri, fh := range s.files {
		// Don't try to reload metadata for go.mod files.
		if fh.Kind() != source.Go {
			continue
		}
		// If the URI doesn't belong to this view, then it's not in a workspace
		// package and should not be reloaded directly.
		if !contains(s.view.session.viewsOf(uri), s.view) {
			continue
		}
		// If the file is not open and is in a vendor directory, don't treat it
		// like a workspace package.
		if _, ok := fh.(*overlay); !ok && inVendor(uri) {
			continue
		}
		// Don't reload metadata for files we've already deemed unloadable.
		if _, ok := s.unloadableFiles[uri]; ok {
			continue
		}
		if s.noValidMetadataForURILocked(uri) {
			files = append(files, fh)
		}
	}
	return files
}

func contains(views []*View, view *View) bool {
	for _, v := range views {
		if v == view {
			return true
		}
	}
	return false
}

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

func generationName(v *View, snapshotID uint64) string {
	return fmt.Sprintf("v%v/%v", v.id, snapshotID)
}

// checkSnapshotLocked verifies that some invariants are preserved on the
// snapshot.
func checkSnapshotLocked(ctx context.Context, s *snapshot) {
	// Check that every go file for a workspace package is identified as
	// belonging to that workspace package.
	for wsID := range s.workspacePackages {
		if m, ok := s.metadata[wsID]; ok {
			for _, uri := range m.GoFiles {
				found := false
				for _, id := range s.ids[uri] {
					if id == wsID {
						found = true
						break
					}
				}
				if !found {
					log.Error.Logf(ctx, "workspace package %v not associated with %v", wsID, uri)
				}
			}
		}
	}
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

func (s *snapshot) clone(ctx, bgCtx context.Context, changes map[span.URI]*fileChange, forceReloadMetadata bool) (*snapshot, bool) {
	var vendorChanged bool
	newWorkspace, workspaceChanged, workspaceReload := s.workspace.invalidate(ctx, changes, &unappliedChanges{
		originalSnapshot: s,
		changes:          changes,
	})

	s.mu.Lock()
	defer s.mu.Unlock()

	checkSnapshotLocked(ctx, s)

	newGen := s.view.session.cache.store.Generation(generationName(s.view, s.id+1))
	bgCtx, cancel := context.WithCancel(bgCtx)
	result := &snapshot{
		id:                s.id + 1,
		generation:        newGen,
		view:              s.view,
		backgroundCtx:     bgCtx,
		cancel:            cancel,
		builtin:           s.builtin,
		initializeOnce:    s.initializeOnce,
		initializedErr:    s.initializedErr,
		ids:               make(map[span.URI][]PackageID, len(s.ids)),
		importedBy:        make(map[PackageID][]PackageID, len(s.importedBy)),
		metadata:          make(map[PackageID]*KnownMetadata, len(s.metadata)),
		packages:          make(map[packageKey]*packageHandle, len(s.packages)),
		actions:           make(map[actionKey]*actionHandle, len(s.actions)),
		files:             make(map[span.URI]source.VersionedFileHandle, len(s.files)),
		goFiles:           make(map[parseKey]*parseGoHandle, len(s.goFiles)),
		symbols:           make(map[span.URI]*symbolHandle, len(s.symbols)),
		workspacePackages: make(map[PackageID]PackagePath, len(s.workspacePackages)),
		unloadableFiles:   make(map[span.URI]struct{}, len(s.unloadableFiles)),
		parseModHandles:   make(map[span.URI]*parseModHandle, len(s.parseModHandles)),
		modTidyHandles:    make(map[span.URI]*modTidyHandle, len(s.modTidyHandles)),
		modWhyHandles:     make(map[span.URI]*modWhyHandle, len(s.modWhyHandles)),
		knownSubdirs:      make(map[span.URI]struct{}, len(s.knownSubdirs)),
		workspace:         newWorkspace,
	}

	if !workspaceChanged && s.workspaceDirHandle != nil {
		result.workspaceDirHandle = s.workspaceDirHandle
		newGen.Inherit(s.workspaceDirHandle)
	}

	// Copy all of the FileHandles.
	for k, v := range s.files {
		result.files[k] = v
	}
	for k, v := range s.symbols {
		if change, ok := changes[k]; ok {
			if change.exists {
				result.symbols[k] = result.buildSymbolHandle(ctx, change.fileHandle)
			}
			continue
		}
		newGen.Inherit(v.handle)
		result.symbols[k] = v
	}

	// Copy the set of unloadable files.
	for k, v := range s.unloadableFiles {
		result.unloadableFiles[k] = v
	}
	// Copy all of the modHandles.
	for k, v := range s.parseModHandles {
		result.parseModHandles[k] = v
	}

	for k, v := range s.goFiles {
		if _, ok := changes[k.file.URI]; ok {
			continue
		}
		newGen.Inherit(v.handle)
		result.goFiles[k] = v
	}

	// Copy all of the go.mod-related handles. They may be invalidated later,
	// so we inherit them at the end of the function.
	for k, v := range s.modTidyHandles {
		if _, ok := changes[k]; ok {
			continue
		}
		result.modTidyHandles[k] = v
	}
	for k, v := range s.modWhyHandles {
		if _, ok := changes[k]; ok {
			continue
		}
		result.modWhyHandles[k] = v
	}

	// Add all of the known subdirectories, but don't update them for the
	// changed files. We need to rebuild the workspace module to know the
	// true set of known subdirectories, but we don't want to do that in clone.
	for k, v := range s.knownSubdirs {
		result.knownSubdirs[k] = v
	}
	for _, c := range changes {
		result.unprocessedSubdirChanges = append(result.unprocessedSubdirChanges, c)
	}

	// directIDs keeps track of package IDs that have directly changed.
	// It maps id->invalidateMetadata.
	directIDs := map[PackageID]bool{}

	// Invalidate all package metadata if the workspace module has changed.
	if workspaceReload {
		for k := range s.metadata {
			directIDs[k] = true
		}
	}

	changedPkgNames := map[PackageID]struct{}{}
	anyImportDeleted := false
	for uri, change := range changes {
		// Maybe reinitialize the view if we see a change in the vendor
		// directory.
		if inVendor(uri) {
			vendorChanged = true
		}

		// The original FileHandle for this URI is cached on the snapshot.
		originalFH := s.files[uri]

		// Check if the file's package name or imports have changed,
		// and if so, invalidate this file's packages' metadata.
		var shouldInvalidateMetadata, pkgNameChanged, importDeleted bool
		if !isGoMod(uri) {
			shouldInvalidateMetadata, pkgNameChanged, importDeleted = s.shouldInvalidateMetadata(ctx, result, originalFH, change.fileHandle)
		}
		invalidateMetadata := forceReloadMetadata || workspaceReload || shouldInvalidateMetadata
		anyImportDeleted = anyImportDeleted || importDeleted

		// Mark all of the package IDs containing the given file.
		// TODO: if the file has moved into a new package, we should invalidate that too.
		filePackageIDs := guessPackageIDsForURI(uri, s.ids)
		if pkgNameChanged {
			for _, id := range filePackageIDs {
				changedPkgNames[id] = struct{}{}
			}
		}
		for _, id := range filePackageIDs {
			directIDs[id] = directIDs[id] || invalidateMetadata
		}

		// Invalidate the previous modTidyHandle if any of the files have been
		// saved or if any of the metadata has been invalidated.
		if invalidateMetadata || fileWasSaved(originalFH, change.fileHandle) {
			// TODO(rstambler): Only delete mod handles for which the
			// withoutURI is relevant.
			for k := range s.modTidyHandles {
				delete(result.modTidyHandles, k)
			}
			for k := range s.modWhyHandles {
				delete(result.modWhyHandles, k)
			}
		}
		if isGoMod(uri) {
			delete(result.parseModHandles, uri)
		}
		// Handle the invalidated file; it may have new contents or not exist.
		if !change.exists {
			delete(result.files, uri)
		} else {
			result.files[uri] = change.fileHandle
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
		for id, metadata := range s.metadata {
			if len(metadata.Errors) > 0 {
				directIDs[id] = true
			}
		}
	}

	// Invalidate reverse dependencies too.
	// TODO(heschi): figure out the locking model and use transitiveReverseDeps?
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
		for _, rid := range s.getImportedByLocked(id) {
			addRevDeps(rid, invalidateMetadata)
		}
	}
	for id, invalidateMetadata := range directIDs {
		addRevDeps(id, invalidateMetadata)
	}

	// Copy the package type information.
	for k, v := range s.packages {
		if _, ok := idsToInvalidate[k.id]; ok {
			continue
		}
		newGen.Inherit(v.handle)
		result.packages[k] = v
	}
	// Copy the package analysis information.
	for k, v := range s.actions {
		if _, ok := idsToInvalidate[k.pkg.id]; ok {
			continue
		}
		newGen.Inherit(v.handle)
		result.actions[k] = v
	}

	// If the workspace mode has changed, we must delete all metadata, as it
	// is unusable and may produce confusing or incorrect diagnostics.
	// If a file has been deleted, we must delete metadata all packages
	// containing that file.
	workspaceModeChanged := s.workspaceMode() != result.workspaceMode()
	skipID := map[PackageID]bool{}
	for _, c := range changes {
		if c.exists {
			continue
		}
		// The file has been deleted.
		if ids, ok := s.ids[c.fileHandle.URI()]; ok {
			for _, id := range ids {
				skipID[id] = true
			}
		}
	}

	// Collect all of the IDs that are reachable from the workspace packages.
	// Any unreachable IDs will have their metadata deleted outright.
	reachableID := map[PackageID]bool{}
	var addForwardDeps func(PackageID)
	addForwardDeps = func(id PackageID) {
		if reachableID[id] {
			return
		}
		reachableID[id] = true
		m, ok := s.metadata[id]
		if !ok {
			return
		}
		for _, depID := range m.Deps {
			addForwardDeps(depID)
		}
	}
	for id := range s.workspacePackages {
		addForwardDeps(id)
	}

	// Copy the URI to package ID mappings, skipping only those URIs whose
	// metadata will be reloaded in future calls to load.
	deleteInvalidMetadata := forceReloadMetadata || workspaceModeChanged
	idsInSnapshot := map[PackageID]bool{} // track all known IDs
	for uri, ids := range s.ids {
		for _, id := range ids {
			invalidateMetadata := idsToInvalidate[id]
			if skipID[id] || (invalidateMetadata && deleteInvalidMetadata) {
				continue
			}
			// The ID is not reachable from any workspace package, so it should
			// be deleted.
			if !reachableID[id] {
				continue
			}
			idsInSnapshot[id] = true
			result.ids[uri] = append(result.ids[uri], id)
		}
	}

	// Copy the package metadata. We only need to invalidate packages directly
	// containing the affected file, and only if it changed in a relevant way.
	for k, v := range s.metadata {
		if !idsInSnapshot[k] {
			// Delete metadata for IDs that are no longer reachable from files
			// in the snapshot.
			continue
		}
		invalidateMetadata := idsToInvalidate[k]
		// Mark invalidated metadata rather than deleting it outright.
		result.metadata[k] = &KnownMetadata{
			Metadata:   v.Metadata,
			Valid:      v.Valid && !invalidateMetadata,
			ShouldLoad: v.ShouldLoad || invalidateMetadata,
		}
	}

	// Copy the set of initially loaded packages.
	for id, pkgPath := range s.workspacePackages {
		// Packages with the id "command-line-arguments" are generated by the
		// go command when the user is outside of GOPATH and outside of a
		// module. Do not cache them as workspace packages for longer than
		// necessary.
		if source.IsCommandLineArguments(string(id)) {
			if invalidateMetadata, ok := idsToInvalidate[id]; invalidateMetadata && ok {
				continue
			}
		}

		// If all the files we know about in a package have been deleted,
		// the package is gone and we should no longer try to load it.
		if m := s.metadata[id]; m != nil {
			hasFiles := false
			for _, uri := range s.metadata[id].GoFiles {
				// For internal tests, we need _test files, not just the normal
				// ones. External tests only have _test files, but we can check
				// them anyway.
				if m.ForTest != "" && !strings.HasSuffix(string(uri), "_test.go") {
					continue
				}
				if _, ok := result.files[uri]; ok {
					hasFiles = true
					break
				}
			}
			if !hasFiles {
				continue
			}
		}

		// If the package name of a file in the package has changed, it's
		// possible that the package ID may no longer exist. Delete it from
		// the set of workspace packages, on the assumption that we will add it
		// back when the relevant files are reloaded.
		if _, ok := changedPkgNames[id]; ok {
			continue
		}

		result.workspacePackages[id] = pkgPath
	}

	// Inherit all of the go.mod-related handles.
	for _, v := range result.modTidyHandles {
		newGen.Inherit(v.handle)
	}
	for _, v := range result.modWhyHandles {
		newGen.Inherit(v.handle)
	}
	for _, v := range result.parseModHandles {
		newGen.Inherit(v.handle)
	}
	// Don't bother copying the importedBy graph,
	// as it changes each time we update metadata.

	// If the snapshot's workspace mode has changed, the packages loaded using
	// the previous mode are no longer relevant, so clear them out.
	if workspaceModeChanged {
		result.workspacePackages = map[PackageID]PackagePath{}
	}

	// The snapshot may need to be reinitialized.
	if workspaceReload || vendorChanged {
		if workspaceChanged || result.initializedErr != nil {
			result.initializeOnce = &sync.Once{}
		}
	}
	return result, workspaceChanged
}

// guessPackageIDsForURI returns all packages related to uri. If we haven't
// seen this URI before, we guess based on files in the same directory. This
// is of course incorrect in build systems where packages are not organized by
// directory.
func guessPackageIDsForURI(uri span.URI, known map[span.URI][]PackageID) []PackageID {
	packages := known[uri]
	if len(packages) > 0 {
		// We've seen this file before.
		return packages
	}
	// This is a file we don't yet know about. Guess relevant packages by
	// considering files in the same directory.

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
	if err != nil {
		return nil
	}

	// Aggregate all possibly relevant package IDs.
	var found []PackageID
	for knownURI, ids := range known {
		knownDir := filepath.Dir(knownURI.Filename())
		knownFI, err := getInfo(knownDir)
		if err != nil {
			continue
		}
		if os.SameFile(fi, knownFI) {
			found = append(found, ids...)
		}
	}
	return found
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

// shouldInvalidateMetadata reparses the full file's AST to determine
// if the file requires a metadata reload.
func (s *snapshot) shouldInvalidateMetadata(ctx context.Context, newSnapshot *snapshot, originalFH, currentFH source.FileHandle) (invalidate, pkgNameChanged, importDeleted bool) {
	if originalFH == nil {
		return true, false, false
	}
	// If the file hasn't changed, there's no need to reload.
	if originalFH.FileIdentity() == currentFH.FileIdentity() {
		return false, false, false
	}
	// Get the original and current parsed files in order to check package name
	// and imports. Use the new snapshot to parse to avoid modifying the
	// current snapshot.
	original, originalErr := newSnapshot.ParseGo(ctx, originalFH, source.ParseFull)
	current, currentErr := newSnapshot.ParseGo(ctx, currentFH, source.ParseFull)
	if originalErr != nil || currentErr != nil {
		return (originalErr == nil) != (currentErr == nil), false, (currentErr != nil) // we don't know if an import was deleted
	}
	// Check if the package's metadata has changed. The cases handled are:
	//    1. A package's name has changed
	//    2. A file's imports have changed
	if original.File.Name.Name != current.File.Name.Name {
		invalidate = true
		pkgNameChanged = true
	}
	origImportSet := make(map[string]struct{})
	for _, importSpec := range original.File.Imports {
		origImportSet[importSpec.Path.Value] = struct{}{}
	}
	curImportSet := make(map[string]struct{})
	for _, importSpec := range current.File.Imports {
		curImportSet[importSpec.Path.Value] = struct{}{}
	}
	// If any of the current imports were not in the original imports.
	for path := range curImportSet {
		if _, ok := origImportSet[path]; ok {
			delete(origImportSet, path)
			continue
		}
		// If the import path is obviously not valid, we can skip reloading
		// metadata. For now, valid means properly quoted and without a
		// terminal slash.
		if isBadImportPath(path) {
			continue
		}
		invalidate = true
	}

	for path := range origImportSet {
		if !isBadImportPath(path) {
			invalidate = true
			importDeleted = true
		}
	}

	if !invalidate {
		invalidate = magicCommentsChanged(original.File, current.File)
	}
	return invalidate, pkgNameChanged, importDeleted
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

func isBadImportPath(path string) bool {
	path, err := strconv.Unquote(path)
	if err != nil {
		return true
	}
	if path == "" {
		return true
	}
	if path[len(path)-1] == '/' {
		return true
	}
	return false
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
		return nil, errors.Errorf("no builtin package for view %s", s.view.name)
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

// TODO(rfindley): move this to workspacemodule.go
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
		if parsed.Go != nil && semver.Compare(goVersion, parsed.Go.Version) < 0 {
			goVersion = parsed.Go.Version
		}
		path := parsed.Module.Mod.Path
		if _, ok := paths[path]; ok {
			return nil, fmt.Errorf("found module %q twice in the workspace", path)
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
			return nil, errors.Errorf("reading go sum: %w", err)
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

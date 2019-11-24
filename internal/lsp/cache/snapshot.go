package cache

import (
	"context"
	"os"
	"sync"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/lsp/telemetry"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/telemetry/log"
	errors "golang.org/x/xerrors"
)

type snapshot struct {
	id   uint64
	view *view

	mu sync.Mutex

	// ids maps file URIs to package IDs.
	// It may be invalidated on calls to go/packages.
	ids map[span.URI][]packageID

	// metadata maps file IDs to their associated metadata.
	// It may invalidated on calls to go/packages.
	metadata map[packageID]*metadata

	// importedBy maps package IDs to the list of packages that import them.
	importedBy map[packageID][]packageID

	// files maps file URIs to their corresponding FileHandles.
	// It may invalidated when a file's content changes.
	files map[span.URI]source.FileHandle

	// packages maps a packageKey to a set of CheckPackageHandles to which that file belongs.
	// It may be invalidated when a file's content changes.
	packages map[packageKey]*checkPackageHandle

	// actions maps an actionkey to its actionHandle.
	actions map[actionKey]*actionHandle

	// workspacePackages contains the workspace's packages, which are loaded
	// when the view is created.
	workspacePackages map[packageID]bool
}

type packageKey struct {
	mode source.ParseMode
	id   packageID
}

type actionKey struct {
	pkg      packageKey
	analyzer *analysis.Analyzer
}

func (s *snapshot) View() source.View {
	return s.view
}

func (s *snapshot) PackageHandles(ctx context.Context, f source.File) ([]source.CheckPackageHandle, error) {
	ctx = telemetry.File.With(ctx, f.URI())

	fh := s.Handle(ctx, f)
	metadata := s.getMetadataForURI(fh.Identity().URI)

	// Determine if we need to type-check the package.
	cphs, load, check := s.shouldCheck(metadata)

	// We may need to re-load package metadata.
	// We only need to this if it has been invalidated, and is therefore unvailable.
	if load {
		var err error
		m, err := s.load(ctx, source.FileURI(f.URI()))
		if err != nil {
			return nil, err
		}
		// If metadata was returned, from the load call, use it.
		if m != nil {
			metadata = m
		}
	}
	if check {
		var results []source.CheckPackageHandle
		for _, m := range metadata {
			cph, err := s.checkPackageHandle(ctx, m.id, source.ParseFull)
			if err != nil {
				return nil, err
			}
			results = append(results, cph)
		}
		cphs = results
	}
	if len(cphs) == 0 {
		return nil, errors.Errorf("no CheckPackageHandles for %s", f.URI())
	}
	return cphs, nil
}

func (s *snapshot) PackageHandle(ctx context.Context, id string) (source.CheckPackageHandle, error) {
	ctx = telemetry.Package.With(ctx, id)

	m := s.getMetadata(packageID(id))
	if m == nil {
		return nil, errors.Errorf("no known metadata for %s", id)
	}
	// Determine if we need to type-check the package.
	cphs, load, check := s.shouldCheck([]*metadata{m})
	if load {
		return nil, errors.Errorf("outdated metadata for %s, needs re-load", id)
	}
	if check {
		return s.checkPackageHandle(ctx, m.id, source.ParseFull)
	}
	if len(cphs) == 0 {
		return nil, errors.Errorf("no check package handle for %s", id)
	}
	if len(cphs) > 1 {
		return nil, errors.Errorf("multiple check package handles for a single id: %s", id)
	}
	return cphs[0], nil
}

// shouldCheck determines if the packages provided by the metadata
// need to be re-loaded or re-type-checked.
func (s *snapshot) shouldCheck(m []*metadata) (cphs []source.CheckPackageHandle, load, check bool) {
	// No metadata. Re-load and re-check.
	if len(m) == 0 {
		return nil, true, true
	}
	// We expect to see a checked package for each package ID,
	// and it should be parsed in full mode.
	// If a single CheckPackageHandle is missing, re-check all of them.
	// TODO: Optimize this by only checking the necessary packages.
	for _, metadata := range m {
		cph := s.getPackage(metadata.id, source.ParseFull)
		if cph == nil {
			return nil, false, true
		}
		cphs = append(cphs, cph)
	}
	// If the metadata for the package had missing dependencies,
	// we _may_ need to re-check. If the missing dependencies haven't changed
	// since previous load, we will not check again.
	if len(cphs) < len(m) {
		for _, m := range m {
			if len(m.missingDeps) != 0 {
				return nil, true, true
			}
		}
	}
	return cphs, false, false
}

func (s *snapshot) GetReverseDependencies(id string) []string {
	ids := make(map[packageID]struct{})
	s.transitiveReverseDependencies(packageID(id), ids)

	// Make sure to delete the original package ID from the map.
	delete(ids, packageID(id))

	var results []string
	for id := range ids {
		results = append(results, string(id))
	}
	return results
}

// transitiveReverseDependencies populates the uris map with file URIs
// belonging to the provided package and its transitive reverse dependencies.
func (s *snapshot) transitiveReverseDependencies(id packageID, ids map[packageID]struct{}) {
	if _, ok := ids[id]; ok {
		return
	}
	m := s.getMetadata(id)
	if m == nil {
		return
	}
	ids[id] = struct{}{}
	importedBy := s.getImportedBy(id)
	for _, parentID := range importedBy {
		s.transitiveReverseDependencies(parentID, ids)
	}
}

func (s *snapshot) getImportedBy(id packageID) []packageID {
	s.mu.Lock()
	defer s.mu.Unlock()

	// If we haven't rebuilt the import graph since creating the snapshot.
	if len(s.importedBy) == 0 {
		s.rebuildImportGraph()
	}

	return s.importedBy[id]
}

func (s *snapshot) addPackage(cph *checkPackageHandle) {
	s.mu.Lock()
	defer s.mu.Unlock()

	// TODO: We should make sure not to compute duplicate CheckPackageHandles,
	// and instead panic here. This will be hard to do because we may encounter
	// the same package multiple times in the dependency tree.
	if _, ok := s.packages[cph.packageKey()]; ok {
		return
	}
	s.packages[cph.packageKey()] = cph
}

// checkWorkspacePackages checks the initial set of packages loaded when
// the view is created. This is needed because
// (*snapshot).CheckPackageHandle makes the assumption that every package that's
// been loaded has an existing checkPackageHandle.
func (s *snapshot) checkWorkspacePackages(ctx context.Context, m []*metadata) ([]source.CheckPackageHandle, error) {
	var cphs []source.CheckPackageHandle
	for _, m := range m {
		cph, err := s.checkPackageHandle(ctx, m.id, source.ParseFull)
		if err != nil {
			return nil, err
		}
		s.workspacePackages[m.id] = true
		cphs = append(cphs, cph)
	}
	return cphs, nil
}

func (s *snapshot) KnownPackages(ctx context.Context) []source.Package {
	// TODO(matloob): This function exists because KnownImportPaths can't
	// determine the import paths of all packages. Remove this function
	// if KnownImportPaths gains that ability. That could happen if
	// go list or go packages provide that information.
	pkgIDs := make(map[packageID]bool)
	s.mu.Lock()
	for _, m := range s.metadata {
		pkgIDs[m.id] = true
	}
	// Add in all the workspacePackages in case the've been invalidated
	// in the metadata since their initial load.
	for id := range s.workspacePackages {
		pkgIDs[id] = true
	}
	s.mu.Unlock()

	var results []source.Package
	for pkgID := range pkgIDs {
		mode := source.ParseExported
		if s.workspacePackages[pkgID] {
			// Any package in our workspace should be loaded with ParseFull.
			mode = source.ParseFull
		}
		cph, err := s.checkPackageHandle(ctx, pkgID, mode)
		if err != nil {
			log.Error(ctx, "failed to create CheckPackageHandle", err, telemetry.Package.Of(pkgID))
			continue
		}
		// Check the package now if it's not checked yet.
		// TODO(matloob): is this too slow?
		pkg, err := cph.check(ctx)
		if err != nil {
			log.Error(ctx, "failed to check package", err, telemetry.Package.Of(pkgID))
			continue
		}
		results = append(results, pkg)
	}

	return results
}

func (s *snapshot) KnownImportPaths() map[string]source.Package {
	s.mu.Lock()
	defer s.mu.Unlock()

	results := map[string]source.Package{}
	for _, cph := range s.packages {
		cachedPkg, err := cph.cached()
		if err != nil {
			continue
		}
		for importPath, newPkg := range cachedPkg.imports {
			if oldPkg, ok := results[string(importPath)]; ok {
				// Using the same trick as NarrowestPackageHandle, prefer non-variants.
				if len(newPkg.compiledGoFiles) < len(oldPkg.(*pkg).compiledGoFiles) {
					results[string(importPath)] = newPkg
				}
			} else {
				results[string(importPath)] = newPkg
			}
		}
	}
	return results
}

func (s *snapshot) getPackage(id packageID, m source.ParseMode) *checkPackageHandle {
	s.mu.Lock()
	defer s.mu.Unlock()

	key := packageKey{
		id:   id,
		mode: m,
	}
	return s.packages[key]
}

func (s *snapshot) getActionHandles(id packageID, m source.ParseMode) []*actionHandle {
	s.mu.Lock()
	defer s.mu.Unlock()

	var acts []*actionHandle
	for k, v := range s.actions {
		if k.pkg.id == id && k.pkg.mode == m {
			acts = append(acts, v)
		}
	}
	return acts
}

func (s *snapshot) getAction(id packageID, m source.ParseMode, a *analysis.Analyzer) *actionHandle {
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

func (s *snapshot) addAction(ah *actionHandle) {
	s.mu.Lock()
	defer s.mu.Unlock()

	key := actionKey{
		analyzer: ah.analyzer,
		pkg: packageKey{
			id:   ah.pkg.id,
			mode: ah.pkg.mode,
		},
	}
	if _, ok := s.actions[key]; ok {
		return
	}
	s.actions[key] = ah
}

func (s *snapshot) getMetadataForURI(uri span.URI) (metadata []*metadata) {
	// TODO(matloob): uri can be a file or directory. Should we update the mappings
	// to map directories to their contained packages?
	s.mu.Lock()
	defer s.mu.Unlock()

	for _, id := range s.ids[uri] {
		if m, ok := s.metadata[id]; ok {
			metadata = append(metadata, m)
		}
	}
	return metadata
}

func (s *snapshot) setMetadata(m *metadata) {
	s.mu.Lock()
	defer s.mu.Unlock()

	// TODO: We should make sure not to set duplicate metadata,
	// and instead panic here. This can be done by making sure not to
	// reset metadata information for packages we've already seen.
	if _, ok := s.metadata[m.id]; ok {
		return
	}
	s.metadata[m.id] = m
}

func (s *snapshot) getMetadata(id packageID) *metadata {
	s.mu.Lock()
	defer s.mu.Unlock()

	return s.metadata[id]
}

func (s *snapshot) addID(uri span.URI, id packageID) {
	s.mu.Lock()
	defer s.mu.Unlock()

	for _, existingID := range s.ids[uri] {
		if existingID == id {
			// TODO: We should make sure not to set duplicate IDs,
			// and instead panic here. This can be done by making sure not to
			// reset metadata information for packages we've already seen.
			return
		}
	}
	s.ids[uri] = append(s.ids[uri], id)
}

func (s *snapshot) getIDs(uri span.URI) []packageID {
	s.mu.Lock()
	defer s.mu.Unlock()

	return s.ids[uri]
}

func (s *snapshot) getFile(uri span.URI) source.FileHandle {
	s.mu.Lock()
	defer s.mu.Unlock()

	return s.files[uri]
}

func (s *snapshot) Handle(ctx context.Context, f source.File) source.FileHandle {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, ok := s.files[f.URI()]; !ok {
		s.files[f.URI()] = s.view.session.GetFile(f.URI(), f.Kind())
	}
	return s.files[f.URI()]
}

func (s *snapshot) clone(ctx context.Context, withoutURI *span.URI, withoutTypes, withoutMetadata map[span.URI]struct{}) *snapshot {
	s.mu.Lock()
	defer s.mu.Unlock()

	result := &snapshot{
		id:                s.id + 1,
		view:              s.view,
		ids:               make(map[span.URI][]packageID),
		importedBy:        make(map[packageID][]packageID),
		metadata:          make(map[packageID]*metadata),
		packages:          make(map[packageKey]*checkPackageHandle),
		actions:           make(map[actionKey]*actionHandle),
		files:             make(map[span.URI]source.FileHandle),
		workspacePackages: make(map[packageID]bool),
	}
	// Copy all of the FileHandles except for the one that was invalidated.
	for k, v := range s.files {
		if withoutURI != nil && k == *withoutURI {
			continue
		}
		result.files[k] = v
	}
	// Collect the IDs for the packages associated with the excluded URIs.
	withoutMetadataIDs := make(map[packageID]struct{})
	withoutTypesIDs := make(map[packageID]struct{})
	for k, ids := range s.ids {
		// Map URIs to IDs for exclusion.
		if withoutTypes != nil {
			if _, ok := withoutTypes[k]; ok {
				for _, id := range ids {
					withoutTypesIDs[id] = struct{}{}
				}
			}
		}
		if withoutMetadata != nil {
			if _, ok := withoutMetadata[k]; ok {
				for _, id := range ids {
					withoutMetadataIDs[id] = struct{}{}
				}
				continue
			}
		}
		result.ids[k] = ids
	}
	// Copy the package type information.
	for k, v := range s.packages {
		if _, ok := withoutTypesIDs[k.id]; ok {
			continue
		}
		if _, ok := withoutMetadataIDs[k.id]; ok {
			continue
		}
		result.packages[k] = v
	}
	// Copy the package analysis information.
	for k, v := range s.actions {
		if _, ok := withoutTypesIDs[k.pkg.id]; ok {
			continue
		}
		if _, ok := withoutMetadataIDs[k.pkg.id]; ok {
			continue
		}
		result.actions[k] = v
	}
	// Copy the package metadata.
	for k, v := range s.metadata {
		if _, ok := withoutMetadataIDs[k]; ok {
			continue
		}
		result.metadata[k] = v
	}
	// Copy the set of initally loaded packages
	for k, v := range s.workspacePackages {
		result.workspacePackages[k] = v
	}
	// Don't bother copying the importedBy graph,
	// as it changes each time we update metadata.
	return result
}

func (s *snapshot) ID() uint64 {
	return s.id
}

// invalidateContent invalidates the content of a Go file,
// including any position and type information that depends on it.
// It returns true if we were already tracking the given file, false otherwise.
func (v *view) invalidateContent(ctx context.Context, f source.File, kind source.FileKind, action source.FileAction) bool {
	// TODO: Handle the possibility of opening a file outside of the current view.
	// For now, return early if we open a file.
	// We assume that we are already tracking any files within the given view.
	if action == source.Open {
		return true
	}
	var (
		withoutTypes    = make(map[span.URI]struct{})
		withoutMetadata = make(map[span.URI]struct{})
		ids             = make(map[packageID]struct{})
	)

	// This should be the only time we hold the view's snapshot lock for any period of time.
	v.snapshotMu.Lock()
	defer v.snapshotMu.Unlock()

	for _, id := range v.snapshot.getIDs(f.URI()) {
		ids[id] = struct{}{}
	}

	// Get the original FileHandle for the URI, if it exists.
	originalFH := v.snapshot.getFile(f.URI())

	switch action {
	case source.Create:
		// If this is a file we don't yet know about,
		// then we do not yet know what packages it should belong to.
		// Make a rough estimate of what metadata to invalidate by finding the package IDs
		// of all of the files in the same directory as this one.
		// TODO(rstambler): Speed this up by mapping directories to filenames.
		if dirStat, err := os.Stat(dir(f.URI().Filename())); err == nil {
			for uri := range v.snapshot.files {
				if fdirStat, err := os.Stat(dir(uri.Filename())); err == nil {
					if os.SameFile(dirStat, fdirStat) {
						for _, id := range v.snapshot.ids[uri] {
							ids[id] = struct{}{}
						}
					}
				}
			}
		}
		originalFH = nil
	}
	if len(ids) == 0 {
		return false
	}

	// Remove the package and all of its reverse dependencies from the cache.
	reverseDependencies := make(map[packageID]struct{})
	for id := range ids {
		v.snapshot.transitiveReverseDependencies(id, reverseDependencies)
	}
	for id := range reverseDependencies {
		m := v.snapshot.getMetadata(id)
		for _, uri := range m.compiledGoFiles {
			withoutTypes[uri] = struct{}{}
		}
	}

	// Make sure to clear out the content if there has been a deletion.
	if action == source.Delete {
		v.session.clearOverlay(f.URI())
	}

	// Get the current FileHandle for the URI.
	currentFH := v.session.GetFile(f.URI(), f.Kind())

	// Check if the file's package name or imports have changed,
	// and if so, invalidate metadata.
	if v.session.cache.shouldLoad(ctx, v.snapshot, originalFH, currentFH) {
		withoutMetadata = withoutTypes

		// TODO: If a package's name has changed,
		// we should invalidate the metadata for the new package name (if it exists).
	}
	uri := f.URI()
	v.snapshot = v.snapshot.clone(ctx, &uri, withoutTypes, withoutMetadata)
	return true
}

func (s *snapshot) clearAndRebuildImportGraph() {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Completely invalidate the original map.
	s.importedBy = make(map[packageID][]packageID)
	s.rebuildImportGraph()
}

func (s *snapshot) rebuildImportGraph() {
	for id, m := range s.metadata {
		for _, importID := range m.deps {
			s.importedBy[importID] = append(s.importedBy[importID], id)
		}
	}
}

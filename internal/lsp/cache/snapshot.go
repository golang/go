// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"context"
	"os"
	"path/filepath"
	"sync"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/lsp/telemetry"
	"golang.org/x/tools/internal/span"
	errors "golang.org/x/xerrors"
)

type snapshot struct {
	id   uint64
	view *view

	// mu guards all of the maps in the snapshot.
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

	// packages maps a packageKey to a set of packageHandles to which that file belongs.
	// It may be invalidated when a file's content changes.
	packages map[packageKey]*packageHandle

	// actions maps an actionkey to its actionHandle.
	actions map[actionKey]*actionHandle

	// workspacePackages contains the workspace's packages, which are loaded
	// when the view is created.
	workspacePackages map[packageID]packagePath
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

func (s *snapshot) ModFiles(ctx context.Context) (source.FileHandle, source.FileHandle, error) {
	r, t, err := s.view.modFiles(ctx)
	if err != nil {
		return nil, nil, err
	}
	if r == "" || t == "" {
		return nil, nil, nil
	}
	// Get the real mod file's content through the snapshot,
	// as it may be open in an overlay.
	realfh, err := s.GetFile(r)
	if err != nil {
		return nil, nil, err
	}
	// Go directly to disk to get the temporary mod file,
	// since it is always on disk.
	tempfh := s.view.session.cache.GetFile(t)
	if tempfh == nil {
		return nil, nil, errors.Errorf("temporary go.mod filehandle is nil")
	}
	return realfh, tempfh, nil
}

func (s *snapshot) PackageHandles(ctx context.Context, fh source.FileHandle) ([]source.PackageHandle, error) {
	// If the file is a go.mod file, go.Packages.Load will always return 0 packages.
	if fh.Identity().Kind == source.Mod {
		return nil, errors.Errorf("attempting to get PackageHandles of .mod file %s", fh.Identity().URI)
	}

	ctx = telemetry.File.With(ctx, fh.Identity().URI)
	meta := s.getMetadataForURI(fh.Identity().URI)

	phs, err := s.packageHandles(ctx, fileURI(fh.Identity().URI), meta)
	if err != nil {
		return nil, err
	}
	var results []source.PackageHandle
	for _, ph := range phs {
		results = append(results, ph)
	}
	return results, nil
}

func (s *snapshot) packageHandle(ctx context.Context, id packageID) (*packageHandle, error) {
	m := s.getMetadata(id)

	// Don't reload metadata in this function.
	// Callers of this function must reload metadata themselves.
	if m == nil {
		return nil, errors.Errorf("%s has no metadata", id)
	}
	phs, load, check := s.shouldCheck([]*metadata{m})
	if load {
		return nil, errors.Errorf("%s needs loading", id)
	}
	if check {
		return s.buildPackageHandle(ctx, m.id, source.ParseFull)
	}
	var result *packageHandle
	for _, ph := range phs {
		if ph.m.id == id {
			if result != nil {
				return nil, errors.Errorf("multiple package handles for the same ID: %s", id)
			}
			result = ph
		}
	}
	if result == nil {
		return nil, errors.Errorf("no PackageHandle for %s", id)
	}
	return result, nil
}

func (s *snapshot) packageHandles(ctx context.Context, scope interface{}, meta []*metadata) ([]*packageHandle, error) {
	// First, determine if we need to reload or recheck the package.
	phs, load, check := s.shouldCheck(meta)
	if load {
		newMeta, err := s.load(ctx, scope)
		if err != nil {
			return nil, err
		}
		newMissing := missingImports(newMeta)
		if len(newMissing) != 0 {
			// Type checking a package with the same missing imports over and over
			// is futile. Don't re-check unless something has changed.
			check = check && !sameSet(missingImports(meta), newMissing)
		}
		meta = newMeta
	}
	var results []*packageHandle
	if check {
		for _, m := range meta {
			ph, err := s.buildPackageHandle(ctx, m.id, source.ParseFull)
			if err != nil {
				return nil, err
			}
			results = append(results, ph)
		}
	} else {
		results = phs
	}
	if len(results) == 0 {
		return nil, errors.Errorf("packageHandles: no package handles for %v", scope)
	}
	return results, nil
}

func missingImports(metadata []*metadata) map[packagePath]struct{} {
	result := map[packagePath]struct{}{}
	for _, m := range metadata {
		for path := range m.missingDeps {
			result[path] = struct{}{}
		}
	}
	return result
}

func sameSet(x, y map[packagePath]struct{}) bool {
	if len(x) != len(y) {
		return false
	}
	for k := range x {
		if _, ok := y[k]; !ok {
			return false
		}
	}
	return true
}

// shouldCheck determines if the packages provided by the metadata
// need to be re-loaded or re-type-checked.
func (s *snapshot) shouldCheck(m []*metadata) (phs []*packageHandle, load, check bool) {
	// No metadata. Re-load and re-check.
	if len(m) == 0 {
		return nil, true, true
	}
	// We expect to see a checked package for each package ID,
	// and it should be parsed in full mode.
	// If a single PackageHandle is missing, re-check all of them.
	// TODO: Optimize this by only checking the necessary packages.
	for _, metadata := range m {
		ph := s.getPackage(metadata.id, source.ParseFull)
		if ph == nil {
			return nil, false, true
		}
		phs = append(phs, ph)
	}
	// If the metadata for the package had missing dependencies,
	// we _may_ need to re-check. If the missing dependencies haven't changed
	// since previous load, we will not check again.
	if len(phs) < len(m) {
		for _, m := range m {
			if len(m.missingDeps) != 0 {
				return nil, true, true
			}
		}
	}
	return phs, false, false
}

func (s *snapshot) GetReverseDependencies(ctx context.Context, id string) ([]source.PackageHandle, error) {
	if err := s.awaitLoaded(ctx); err != nil {
		return nil, err
	}
	ids := make(map[packageID]struct{})
	s.transitiveReverseDependencies(packageID(id), ids)

	// Make sure to delete the original package ID from the map.
	delete(ids, packageID(id))

	var results []source.PackageHandle
	for id := range ids {
		ph, err := s.packageHandle(ctx, id)
		if err != nil {
			return nil, err
		}
		results = append(results, ph)
	}
	return results, nil
}

// transitiveReverseDependencies populates the uris map with file URIs
// belonging to the provided package and its transitive reverse dependencies.
func (s *snapshot) transitiveReverseDependencies(id packageID, ids map[packageID]struct{}) {
	if _, ok := ids[id]; ok {
		return
	}
	if s.getMetadata(id) == nil {
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
	return s.getImportedByLocked(id)
}

func (s *snapshot) getImportedByLocked(id packageID) []packageID {
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

func (s *snapshot) addPackage(ph *packageHandle) {
	s.mu.Lock()
	defer s.mu.Unlock()

	// TODO: We should make sure not to compute duplicate packageHandles,
	// and instead panic here. This will be hard to do because we may encounter
	// the same package multiple times in the dependency tree.
	if _, ok := s.packages[ph.packageKey()]; ok {
		return
	}
	s.packages[ph.packageKey()] = ph
}

func (s *snapshot) workspacePackageIDs() (ids []packageID) {
	s.mu.Lock()
	defer s.mu.Unlock()

	for id := range s.workspacePackages {
		ids = append(ids, id)
	}
	return ids
}

func (s *snapshot) WorkspacePackages(ctx context.Context) ([]source.PackageHandle, error) {
	if err := s.awaitLoaded(ctx); err != nil {
		return nil, err
	}
	var results []source.PackageHandle
	for _, pkgID := range s.workspacePackageIDs() {
		ph, err := s.packageHandle(ctx, pkgID)
		if err != nil {
			return nil, err
		}
		results = append(results, ph)
	}
	return results, nil
}

func (s *snapshot) KnownPackages(ctx context.Context) ([]source.PackageHandle, error) {
	if err := s.awaitLoaded(ctx); err != nil {
		return nil, err
	}
	// Collect PackageHandles for all of the workspace packages first.
	// They may need to be reloaded if their metadata has been invalidated.
	wsPackages := make(map[packageID]bool)
	s.mu.Lock()
	for id := range s.workspacePackages {
		wsPackages[id] = true
	}
	s.mu.Unlock()

	var results []source.PackageHandle
	for pkgID := range wsPackages {
		ph, err := s.packageHandle(ctx, pkgID)
		if err != nil {
			return nil, err
		}
		results = append(results, ph)
	}

	// Once all workspace packages have been checked, the metadata will be up-to-date.
	// Add all packages known in the workspace (that haven't already been added).
	pkgIDs := make(map[packageID]bool)
	s.mu.Lock()
	for id := range s.metadata {
		if !wsPackages[id] {
			pkgIDs[id] = true
		}
	}
	s.mu.Unlock()

	for pkgID := range pkgIDs {
		// Metadata for these packages should already be up-to-date,
		// so just build the package handle directly (without a reload).
		ph, err := s.buildPackageHandle(ctx, pkgID, source.ParseExported)
		if err != nil {
			return nil, err
		}
		results = append(results, ph)
	}
	return results, nil
}

func (s *snapshot) CachedImportPaths(ctx context.Context) (map[string]source.Package, error) {
	// Don't reload workspace package metadata.
	// This function is meant to only return currently cached information.
	if err := s.view.awaitInitialized(ctx); err != nil {
		return nil, err
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	results := map[string]source.Package{}
	for _, ph := range s.packages {
		cachedPkg, err := ph.cached()
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
	return results, nil
}

func (s *snapshot) getPackage(id packageID, m source.ParseMode) *packageHandle {
	s.mu.Lock()
	defer s.mu.Unlock()

	key := packageKey{
		id:   id,
		mode: m,
	}
	return s.packages[key]
}

func (s *snapshot) getActionHandle(id packageID, m source.ParseMode, a *analysis.Analyzer) *actionHandle {
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

func (s *snapshot) addActionHandle(ah *actionHandle) {
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

func (s *snapshot) isWorkspacePackage(id packageID) (packagePath, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()

	scope, ok := s.workspacePackages[id]
	return scope, ok
}

func (s *snapshot) setWorkspacePackage(id packageID, pkgPath packagePath) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.workspacePackages[id] = pkgPath
}

func (s *snapshot) getFileURIs() []span.URI {
	s.mu.Lock()
	defer s.mu.Unlock()

	var uris []span.URI
	for uri := range s.files {
		uris = append(uris, uri)
	}
	return uris
}

// GetFile returns a File for the given URI. It will always succeed because it
// adds the file to the managed set if needed.
func (s *snapshot) GetFile(uri span.URI) (source.FileHandle, error) {
	f, err := s.view.getFileLocked(uri)
	if err != nil {
		return nil, err
	}
	return s.getFileHandle(f), nil
}

func (s *snapshot) getFileHandle(f *fileBase) source.FileHandle {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, ok := s.files[f.URI()]; !ok {
		s.files[f.URI()] = s.view.session.GetFile(f.URI())
	}
	return s.files[f.URI()]
}

func (s *snapshot) findFileHandle(f *fileBase) source.FileHandle {
	s.mu.Lock()
	defer s.mu.Unlock()

	return s.files[f.URI()]
}

func (s *snapshot) awaitLoaded(ctx context.Context) error {
	// Do not return results until the snapshot's view has been initialized.
	if err := s.view.awaitInitialized(ctx); err != nil {
		return err
	}
	// Make sure that the workspace is in a valid state.
	return s.reloadWorkspace(ctx)
}

// reloadWorkspace reloads the metadata for all invalidated workspace packages.
func (s *snapshot) reloadWorkspace(ctx context.Context) error {
	scope := s.workspaceScope(ctx)
	if scope == nil {
		return nil
	}
	_, err := s.load(ctx, scope)
	return err
}

func (s *snapshot) workspaceScope(ctx context.Context) interface{} {
	s.mu.Lock()
	defer s.mu.Unlock()

	var pkgPaths []packagePath
	for id, pkgPath := range s.workspacePackages {
		if s.metadata[id] == nil {
			pkgPaths = append(pkgPaths, pkgPath)
		}
	}
	switch len(pkgPaths) {
	case 0:
		return nil
	case len(s.workspacePackages):
		return directoryURI(s.view.folder)
	default:
		return pkgPaths
	}
}

func (s *snapshot) clone(ctx context.Context, withoutURI span.URI) *snapshot {
	s.mu.Lock()
	defer s.mu.Unlock()

	directIDs := map[packageID]struct{}{}
	// Collect all of the package IDs that correspond to the given file.
	// TODO: if the file has moved into a new package, we should invalidate that too.
	for _, id := range s.ids[withoutURI] {
		directIDs[id] = struct{}{}
	}

	// Get the current and original FileHandles for this URI.
	currentFH := s.view.session.GetFile(withoutURI)
	originalFH := s.files[withoutURI]

	// Check if the file's package name or imports have changed,
	// and if so, invalidate this file's packages' metadata.
	invalidateMetadata := s.view.session.cache.shouldLoad(ctx, s, originalFH, currentFH)

	// If a go.mod file's contents have changed, invalidate the metadata
	// for all of the packages in the workspace.
	if invalidateMetadata && currentFH.Identity().Kind == source.Mod {
		for id := range s.workspacePackages {
			directIDs[id] = struct{}{}
		}
	}

	// If this is a file we don't yet know about,
	// then we do not yet know what packages it should belong to.
	// Make a rough estimate of what metadata to invalidate by finding the package IDs
	// of all of the files in the same directory as this one.
	// TODO(rstambler): Speed this up by mapping directories to filenames.
	if len(directIDs) == 0 {
		if dirStat, err := os.Stat(filepath.Dir(withoutURI.Filename())); err == nil {
			for uri := range s.files {
				if fdirStat, err := os.Stat(filepath.Dir(uri.Filename())); err == nil {
					if os.SameFile(dirStat, fdirStat) {
						for _, id := range s.ids[uri] {
							directIDs[id] = struct{}{}
						}
					}
				}
			}
		}
	}

	// Invalidate reverse dependencies too.
	// TODO(heschi): figure out the locking model and use transitiveReverseDeps?
	transitiveIDs := make(map[packageID]struct{})
	var addRevDeps func(packageID)
	addRevDeps = func(id packageID) {
		if _, seen := transitiveIDs[id]; seen {
			return
		}
		transitiveIDs[id] = struct{}{}
		for _, rid := range s.getImportedByLocked(id) {
			addRevDeps(rid)
		}
	}
	for id := range directIDs {
		addRevDeps(id)
	}
	// Invalidate metadata for the transitive dependencies,
	// if they are x_tests and test variants.
	//
	// An example:
	//
	// The only way to reload the metadata for
	// golang.org/x/tools/internal/lsp/cache [golang.org/x/tools/internal/lsp/source.test]
	// is by reloading golang.org/x/tools/internal/lsp/source.
	// That means we have to invalidate the metadata for
	// golang.org/x/tools/internal/lsp/source_test when invalidating metadata for
	// golang.org/x/tools/internal/lsp/cache.
	for id := range transitiveIDs {
		if m := s.metadata[id]; m != nil && m.forTest != "" {
			directIDs[id] = struct{}{}
		}
	}

	result := &snapshot{
		id:                s.id + 1,
		view:              s.view,
		ids:               make(map[span.URI][]packageID),
		importedBy:        make(map[packageID][]packageID),
		metadata:          make(map[packageID]*metadata),
		packages:          make(map[packageKey]*packageHandle),
		actions:           make(map[actionKey]*actionHandle),
		files:             make(map[span.URI]source.FileHandle),
		workspacePackages: make(map[packageID]packagePath),
	}

	// Copy all of the FileHandles.
	for k, v := range s.files {
		result.files[k] = v
	}

	// Handle the invalidated file; it may have new contents or not exist.
	if _, _, err := currentFH.Read(ctx); os.IsNotExist(err) {
		delete(result.files, withoutURI)
	} else {
		result.files[withoutURI] = currentFH
	}

	// Collect the IDs for the packages associated with the excluded URIs.
	for k, ids := range s.ids {
		result.ids[k] = ids
	}
	// Copy the package type information.
	for k, v := range s.packages {
		if _, ok := transitiveIDs[k.id]; ok {
			continue
		}
		result.packages[k] = v
	}
	// Copy the package analysis information.
	for k, v := range s.actions {
		if _, ok := transitiveIDs[k.pkg.id]; ok {
			continue
		}
		result.actions[k] = v
	}
	// Copy the set of initally loaded packages.
	for k, v := range s.workspacePackages {
		result.workspacePackages[k] = v
	}

	// Copy the package metadata. We only need to invalidate packages directly
	// containing the affected file, and only if it changed in a relevant way.
	for k, v := range s.metadata {
		if _, ok := directIDs[k]; invalidateMetadata && ok {
			continue
		}
		result.metadata[k] = v
	}
	// Don't bother copying the importedBy graph,
	// as it changes each time we update metadata.
	return result
}

func (s *snapshot) ID() uint64 {
	return s.id
}

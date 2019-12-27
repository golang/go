// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

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

	// packages maps a packageKey to a set of CheckPackageHandles to which that file belongs.
	// It may be invalidated when a file's content changes.
	packages map[packageKey]*packageHandle

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

func (s *snapshot) PackageHandle(ctx context.Context, pkgID string) (source.PackageHandle, error) {
	id := packageID(pkgID)
	var meta []*metadata
	if m := s.getMetadata(id); m != nil {
		meta = append(meta, m)
	}
	phs, err := s.packageHandles(ctx, id, meta)
	if err != nil {
		return nil, err
	}
	if len(phs) > 1 {
		return nil, errors.Errorf("more than one package for %s", id)
	}
	return phs[0], nil
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
	// If a single CheckPackageHandle is missing, re-check all of them.
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
	return s.getImportedByLocked(id)
}

func (s *snapshot) getImportedByLocked(id packageID) []packageID {
	// If we haven't rebuilt the import graph since creating the snapshot.
	if len(s.importedBy) == 0 {
		s.rebuildImportGraph()
	}

	return s.importedBy[id]
}

func (s *snapshot) addPackage(ph *packageHandle) {
	s.mu.Lock()
	defer s.mu.Unlock()

	// TODO: We should make sure not to compute duplicate CheckPackageHandles,
	// and instead panic here. This will be hard to do because we may encounter
	// the same package multiple times in the dependency tree.
	if _, ok := s.packages[ph.packageKey()]; ok {
		return
	}
	s.packages[ph.packageKey()] = ph
}

// checkWorkspacePackages checks the initial set of packages loaded when
// the view is created. This is needed because
// (*snapshot).CheckPackageHandle makes the assumption that every package that's
// been loaded has an existing checkPackageHandle.
func (s *snapshot) checkWorkspacePackages(ctx context.Context, m []*metadata) ([]source.PackageHandle, error) {
	var phs []source.PackageHandle
	for _, m := range m {
		ph, err := s.buildPackageHandle(ctx, m.id, source.ParseFull)
		if err != nil {
			return nil, err
		}
		s.workspacePackages[m.id] = true
		phs = append(phs, ph)
	}
	return phs, nil
}

func (s *snapshot) WorkspacePackageIDs(ctx context.Context) (ids []string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	for id := range s.workspacePackages {
		ids = append(ids, string(id))
	}
	return ids
}

func (s *snapshot) KnownPackages(ctx context.Context) []source.PackageHandle {
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
		ph, err := s.PackageHandle(ctx, string(pkgID))
		if err != nil {
			log.Error(ctx, "KnownPackages: failed to create PackageHandle", err, telemetry.Package.Of(pkgID))
			continue
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
			log.Error(ctx, "KnownPackages: failed to create PackageHandle", err, telemetry.Package.Of(pkgID))
			continue
		}
		results = append(results, ph)
	}

	return results
}

func (s *snapshot) KnownImportPaths() map[string]source.Package {
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
	return results
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

func (s *snapshot) getFileURIs() []span.URI {
	s.mu.Lock()
	defer s.mu.Unlock()

	var uris []span.URI
	for uri := range s.files {
		uris = append(uris, uri)
	}
	return uris
}

// FindFile returns the file if the given URI is already a part of the view.
func (s *snapshot) FindFile(ctx context.Context, uri span.URI) source.FileHandle {
	s.view.mu.Lock()
	defer s.view.mu.Unlock()

	f, err := s.view.findFile(uri)
	if f == nil || err != nil {
		return nil
	}
	return s.getFileHandle(ctx, f)
}

// GetFile returns a File for the given URI. It will always succeed because it
// adds the file to the managed set if needed.
func (s *snapshot) GetFile(ctx context.Context, uri span.URI) (source.FileHandle, error) {
	s.view.mu.Lock()
	defer s.view.mu.Unlock()

	// TODO(rstambler): Should there be a version that provides a kind explicitly?
	kind := source.DetectLanguage("", uri.Filename())
	f, err := s.view.getFile(ctx, uri, kind)
	if err != nil {
		return nil, err
	}
	return s.getFileHandle(ctx, f), nil
}

func (s *snapshot) getFileHandle(ctx context.Context, f *fileBase) source.FileHandle {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, ok := s.files[f.URI()]; !ok {
		s.files[f.URI()] = s.view.session.GetFile(f.URI(), f.kind)
	}
	return s.files[f.URI()]
}

func (s *snapshot) clone(ctx context.Context, withoutURI span.URI, withoutFileKind source.FileKind) *snapshot {
	s.mu.Lock()
	defer s.mu.Unlock()

	directIDs := map[packageID]struct{}{}
	// Collect all of the package IDs that correspond to the given file.
	// TODO: if the file has moved into a new package, we should invalidate that too.
	for _, id := range s.ids[withoutURI] {
		directIDs[id] = struct{}{}
	}

	// If we are invalidating a go.mod file then we should invalidate all of the packages in the module
	if withoutFileKind == source.Mod {
		for id := range s.workspacePackages {
			directIDs[id] = struct{}{}
		}
	}

	// Get the original FileHandle for the URI, if it exists.
	originalFH := s.files[withoutURI]

	// If this is a file we don't yet know about,
	// then we do not yet know what packages it should belong to.
	// Make a rough estimate of what metadata to invalidate by finding the package IDs
	// of all of the files in the same directory as this one.
	// TODO(rstambler): Speed this up by mapping directories to filenames.
	if originalFH == nil {
		if dirStat, err := os.Stat(dir(withoutURI.Filename())); err == nil {
			for uri := range s.files {
				if fdirStat, err := os.Stat(dir(uri.Filename())); err == nil {
					if os.SameFile(dirStat, fdirStat) {
						for _, id := range s.ids[uri] {
							directIDs[id] = struct{}{}
						}
					}
				}
			}
		}
	}

	// If there is no known FileHandle and no known IDs for the given file,
	// there is nothing to invalidate.
	if len(directIDs) == 0 && originalFH == nil {
		// TODO(heschi): clone anyway? Seems like this is just setting us up for trouble.
		return s
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

	result := &snapshot{
		id:                s.id + 1,
		view:              s.view,
		ids:               make(map[span.URI][]packageID),
		importedBy:        make(map[packageID][]packageID),
		metadata:          make(map[packageID]*metadata),
		packages:          make(map[packageKey]*packageHandle),
		actions:           make(map[actionKey]*actionHandle),
		files:             make(map[span.URI]source.FileHandle),
		workspacePackages: make(map[packageID]bool),
	}

	// Copy all of the FileHandles.
	for k, v := range s.files {
		result.files[k] = v
	}
	// Handle the invalidated file; it may have new contents or not exist.
	currentFH := s.view.session.GetFile(withoutURI, withoutFileKind)
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

	// Check if the file's package name or imports have changed,
	// and if so, invalidate this file's packages' metadata.
	invalidateMetadata := s.view.session.cache.shouldLoad(ctx, s, originalFH, currentFH)

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

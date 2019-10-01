package cache

import (
	"context"
	"sync"

	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
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

	// packages maps a file URI to a set of CheckPackageHandles to which that file belongs.
	// It may be invalidated when a file's content changes.
	packages map[packageKey]*checkPackageHandle
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

func (s *snapshot) getPackages(uri span.URI, m source.ParseMode) (cphs []source.CheckPackageHandle) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if ids, ok := s.ids[uri]; ok {
		for _, id := range ids {
			key := packageKey{
				id:   id,
				mode: m,
			}
			cph, ok := s.packages[key]
			if ok {
				cphs = append(cphs, cph)
			}
		}
	}
	return cphs
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

func (s *snapshot) getMetadataForURI(uri span.URI) (metadata []*metadata) {
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
		id:         s.id + 1,
		view:       s.view,
		ids:        make(map[span.URI][]packageID),
		importedBy: make(map[packageID][]packageID),
		metadata:   make(map[packageID]*metadata),
		packages:   make(map[packageKey]*checkPackageHandle),
		files:      make(map[span.URI]source.FileHandle),
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
	// Copy the package metadata.
	for k, v := range s.metadata {
		if _, ok := withoutMetadataIDs[k]; ok {
			continue
		}
		result.metadata[k] = v
	}
	// Don't bother copying the importedBy graph,
	// as it changes each time we update metadata.
	return result
}

// invalidateContent invalidates the content of a Go file,
// including any position and type information that depends on it.
func (v *view) invalidateContent(ctx context.Context, uri span.URI, kind source.FileKind) {
	withoutTypes := make(map[span.URI]struct{})
	withoutMetadata := make(map[span.URI]struct{})

	// This should be the only time we hold the view's snapshot lock for any period of time.
	v.snapshotMu.Lock()
	defer v.snapshotMu.Unlock()

	ids := v.snapshot.getIDs(uri)

	// Remove the package and all of its reverse dependencies from the cache.
	for _, id := range ids {
		v.snapshot.reverseDependencies(id, withoutTypes, map[packageID]struct{}{})
	}

	// Get the original FileHandle for the URI, if it exists.
	originalFH := v.snapshot.getFile(uri)

	// Get the current FileHandle for the URI.
	currentFH := v.session.GetFile(uri, kind)

	// Check if the file's package name or imports have changed,
	// and if so, invalidate metadata.
	if v.session.cache.shouldLoad(ctx, v.snapshot, originalFH, currentFH) {
		withoutMetadata = withoutTypes

		// TODO: If a package's name has changed,
		// we should invalidate the metadata for the new package name (if it exists).
	}
	v.snapshot = v.snapshot.clone(ctx, &uri, withoutTypes, withoutMetadata)
}

// invalidateMetadata invalidates package metadata for all files in f's
// package. This forces f's package's metadata to be reloaded next
// time the package is checked.
//
// TODO: This function shouldn't be necessary.
// We should be able to handle its use cases more efficiently.
func (v *view) invalidateMetadata(ctx context.Context, uri span.URI) {
	v.snapshotMu.Lock()
	defer v.snapshotMu.Unlock()

	withoutMetadata := make(map[span.URI]struct{})

	for _, id := range v.snapshot.getIDs(uri) {
		v.snapshot.reverseDependencies(id, withoutMetadata, map[packageID]struct{}{})
	}
	v.snapshot = v.snapshot.clone(ctx, nil, withoutMetadata, withoutMetadata)
}

// reverseDependencies populates the uris map with file URIs belonging to the
// provided package and its transitive reverse dependencies.
func (s *snapshot) reverseDependencies(id packageID, uris map[span.URI]struct{}, seen map[packageID]struct{}) {
	if _, ok := seen[id]; ok {
		return
	}
	m := s.getMetadata(id)
	if m == nil {
		return
	}
	seen[id] = struct{}{}
	importedBy := s.getImportedBy(id)
	for _, parentID := range importedBy {
		s.reverseDependencies(parentID, uris, seen)
	}
	for _, uri := range m.files {
		uris[uri] = struct{}{}
	}
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

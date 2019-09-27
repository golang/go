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
	packages map[span.URI]map[packageKey]*checkPackageHandle
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

func (s *snapshot) addPackage(uri span.URI, cph *checkPackageHandle) {
	s.mu.Lock()
	defer s.mu.Unlock()

	pkgs, ok := s.packages[uri]
	if !ok {
		pkgs = make(map[packageKey]*checkPackageHandle)
		s.packages[uri] = pkgs
	}
	// TODO: Check that there isn't already something set here.
	// This can't be done until we fix the cache keys for CheckPackageHandles.
	pkgs[packageKey{
		id:   cph.m.id,
		mode: cph.Mode(),
	}] = cph
}

func (s *snapshot) getPackages(uri span.URI) (cphs []source.CheckPackageHandle) {
	s.mu.Lock()
	defer s.mu.Unlock()

	for _, cph := range s.packages[uri] {
		cphs = append(cphs, cph)
	}
	return cphs
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

func (s *snapshot) clone(withoutURI span.URI, withoutTypes, withoutMetadata map[span.URI]struct{}) *snapshot {
	s.mu.Lock()
	defer s.mu.Unlock()

	result := &snapshot{
		id:         s.id + 1,
		view:       s.view,
		packages:   make(map[span.URI]map[packageKey]*checkPackageHandle),
		ids:        make(map[span.URI][]packageID),
		metadata:   make(map[packageID]*metadata),
		importedBy: make(map[packageID][]packageID),
		files:      make(map[span.URI]source.FileHandle),
	}
	// Copy all of the FileHandles except for the one that was invalidated.
	for k, v := range s.files {
		if k == withoutURI {
			continue
		}
		result.files[k] = v
	}
	for k, v := range s.packages {
		if withoutTypes != nil {
			if _, ok := withoutTypes[k]; ok {
				continue
			}
		}
		result.packages[k] = v
	}
	withoutIDs := make(map[packageID]struct{})
	for k, ids := range s.ids {
		if withoutMetadata != nil {
			if _, ok := withoutMetadata[k]; ok {
				for _, id := range ids {
					withoutIDs[id] = struct{}{}
				}
				continue
			}
		}
		result.ids[k] = ids
	}
	for k, v := range s.metadata {
		if _, ok := withoutIDs[k]; ok {
			continue
		}
		result.metadata[k] = v
	}
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

	v.snapshot = v.snapshot.clone(uri, withoutTypes, withoutMetadata)
}

// invalidateMetadata invalidates package metadata for all files in f's
// package. This forces f's package's metadata to be reloaded next
// time the package is checked.
//
// TODO: This function shouldn't be necessary.
// We should be able to handle its use cases more efficiently.
func (v *view) invalidateMetadata(uri span.URI) {
	v.snapshotMu.Lock()
	defer v.snapshotMu.Unlock()

	withoutMetadata := make(map[span.URI]struct{})
	for _, id := range v.snapshot.getIDs(uri) {
		v.snapshot.reverseDependencies(id, withoutMetadata, map[packageID]struct{}{})
	}
	v.snapshot = v.snapshot.clone(uri, nil, withoutMetadata)
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

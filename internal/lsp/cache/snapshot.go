package cache

import (
	"context"
	"go/types"

	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/telemetry/log"
	"golang.org/x/tools/internal/telemetry/tag"
	errors "golang.org/x/xerrors"
)

type snapshot struct {
	id uint64

	packages map[span.URI]map[packageKey]*checkPackageHandle
	ids      map[span.URI][]packageID
	metadata map[packageID]*metadata
}

type metadata struct {
	id          packageID
	pkgPath     packagePath
	name        string
	files       []span.URI
	typesSizes  types.Sizes
	parents     map[packageID]bool
	children    map[packageID]*metadata
	errors      []packages.Error
	missingDeps map[packagePath]struct{}
}

func (v *view) getSnapshot(uri span.URI) ([]*metadata, []*checkPackageHandle) {
	v.snapshotMu.Lock()
	defer v.snapshotMu.Unlock()

	var m []*metadata
	for _, id := range v.snapshot.ids[uri] {
		m = append(m, v.snapshot.metadata[id])
	}
	var cphs []*checkPackageHandle
	for _, cph := range v.snapshot.packages[uri] {
		cphs = append(cphs, cph)
	}
	return m, cphs
}

func (v *view) getMetadata(uri span.URI) []*metadata {
	v.snapshotMu.Lock()
	defer v.snapshotMu.Unlock()

	var m []*metadata
	for _, id := range v.snapshot.ids[uri] {
		m = append(m, v.snapshot.metadata[id])
	}
	return m
}

func (v *view) getPackages(uri span.URI) map[packageKey]*checkPackageHandle {
	v.snapshotMu.Lock()
	defer v.snapshotMu.Unlock()

	return v.snapshot.packages[uri]
}

func (v *view) updateMetadata(ctx context.Context, uri span.URI, pkgs []*packages.Package) ([]*metadata, map[packageID]map[packagePath]struct{}, error) {
	v.snapshotMu.Lock()
	defer v.snapshotMu.Unlock()

	// Clear metadata since we are re-running go/packages.
	prevMissingImports := make(map[packageID]map[packagePath]struct{})
	for _, id := range v.snapshot.ids[uri] {
		if m, ok := v.snapshot.metadata[id]; ok && len(m.missingDeps) > 0 {
			prevMissingImports[id] = m.missingDeps
		}
	}
	without := make(map[span.URI]struct{})
	for _, id := range v.snapshot.ids[uri] {
		v.remove(id, without, map[packageID]struct{}{})
	}
	v.snapshot = v.snapshot.cloneMetadata(without)

	var results []*metadata
	for _, pkg := range pkgs {
		log.Print(ctx, "go/packages.Load", tag.Of("package", pkg.PkgPath), tag.Of("files", pkg.CompiledGoFiles))

		// Build the import graph for this package.
		if err := v.updateImportGraph(ctx, &importGraph{
			pkgPath: packagePath(pkg.PkgPath),
			pkg:     pkg,
			parent:  nil,
		}); err != nil {
			return nil, nil, err
		}
		results = append(results, v.snapshot.metadata[packageID(pkg.ID)])
	}
	if len(results) == 0 {
		return nil, nil, errors.Errorf("no metadata for %s", uri)
	}
	return results, prevMissingImports, nil
}

type importGraph struct {
	pkgPath packagePath
	pkg     *packages.Package
	parent  *metadata
}

func (v *view) updateImportGraph(ctx context.Context, g *importGraph) error {
	// Recreate the metadata rather than reusing it to avoid locking.
	m := &metadata{
		id:         packageID(g.pkg.ID),
		pkgPath:    g.pkgPath,
		name:       g.pkg.Name,
		typesSizes: g.pkg.TypesSizes,
		errors:     g.pkg.Errors,
	}
	for _, filename := range g.pkg.CompiledGoFiles {
		uri := span.FileURI(filename)
		v.snapshot.ids[uri] = append(v.snapshot.ids[uri], m.id)
		m.files = append(m.files, uri)
	}
	// Preserve the import graph.
	if original, ok := v.snapshot.metadata[m.id]; ok {
		m.children = original.children
		m.parents = original.parents
	}
	if m.children == nil {
		m.children = make(map[packageID]*metadata)
	}
	if m.parents == nil {
		m.parents = make(map[packageID]bool)
	}

	// Add the metadata to the cache.
	v.snapshot.metadata[m.id] = m

	// Connect the import graph.
	if g.parent != nil {
		m.parents[g.parent.id] = true
		g.parent.children[m.id] = m
	}
	for importPath, importPkg := range g.pkg.Imports {
		importPkgPath := packagePath(importPath)
		if importPkgPath == g.pkgPath {
			return errors.Errorf("cycle detected in %s", importPath)
		}
		// Don't remember any imports with significant errors.
		if importPkgPath != "unsafe" && len(importPkg.CompiledGoFiles) == 0 {
			if m.missingDeps == nil {
				m.missingDeps = make(map[packagePath]struct{})
			}
			m.missingDeps[importPkgPath] = struct{}{}
			continue
		}
		if _, ok := m.children[packageID(importPkg.ID)]; !ok {
			if err := v.updateImportGraph(ctx, &importGraph{
				pkgPath: importPkgPath,
				pkg:     importPkg,
				parent:  m,
			}); err != nil {
				log.Error(ctx, "error in dependency", err)
			}
		}
	}
	// Clear out any imports that have been removed since the package was last loaded.
	for importID := range m.children {
		child, ok := v.snapshot.metadata[importID]
		if !ok {
			continue
		}
		importPath := string(child.pkgPath)
		if _, ok := g.pkg.Imports[importPath]; ok {
			continue
		}
		delete(m.children, importID)
		delete(child.parents, m.id)
	}
	return nil
}

func (v *view) updatePackages(cphs []*checkPackageHandle) {
	v.snapshotMu.Lock()
	defer v.snapshotMu.Unlock()

	for _, cph := range cphs {
		for _, ph := range cph.files {
			uri := ph.File().Identity().URI
			if _, ok := v.snapshot.packages[uri]; !ok {
				v.snapshot.packages[uri] = make(map[packageKey]*checkPackageHandle)
			}
			v.snapshot.packages[uri][packageKey{
				id:   cph.m.id,
				mode: ph.Mode(),
			}] = cph
		}
	}
}

// invalidateContent invalidates the content of a Go file,
// including any position and type information that depends on it.
func (v *view) invalidateContent(ctx context.Context, f *goFile) {
	f.handleMu.Lock()
	defer f.handleMu.Unlock()

	without := make(map[span.URI]struct{})

	// Remove the package and all of its reverse dependencies from the cache.
	v.snapshotMu.Lock()
	defer v.snapshotMu.Unlock()

	for _, id := range v.snapshot.ids[f.URI()] {
		f.view.remove(id, without, map[packageID]struct{}{})
	}
	v.snapshot = v.snapshot.clonePackages(without)
	f.handle = nil
}

// invalidateMeta invalidates package metadata for all files in f's
// package. This forces f's package's metadata to be reloaded next
// time the package is checked.
func (v *view) invalidateMetadata(uri span.URI) {
	v.snapshotMu.Lock()
	defer v.snapshotMu.Unlock()

	without := make(map[span.URI]struct{})

	for _, id := range v.snapshot.ids[uri] {
		v.remove(id, without, map[packageID]struct{}{})
	}
	v.snapshot = v.snapshot.cloneMetadata(without)
}

// remove invalidates a package and its reverse dependencies in the view's
// package cache. It is assumed that the caller has locked both the mutexes
// of both the mcache and the pcache.
func (v *view) remove(id packageID, toDelete map[span.URI]struct{}, seen map[packageID]struct{}) {
	if _, ok := seen[id]; ok {
		return
	}
	m, ok := v.snapshot.metadata[id]
	if !ok {
		return
	}
	seen[id] = struct{}{}
	for parentID := range m.parents {
		v.remove(parentID, toDelete, seen)
	}
	for _, uri := range m.files {
		toDelete[uri] = struct{}{}
	}
}

func (s *snapshot) clonePackages(without map[span.URI]struct{}) *snapshot {
	result := &snapshot{
		id:       s.id + 1,
		packages: make(map[span.URI]map[packageKey]*checkPackageHandle),
		ids:      s.ids,
		metadata: s.metadata,
	}
	for k, v := range s.packages {
		if _, ok := without[k]; ok {
			continue
		}
		result.packages[k] = v
	}
	return result
}

func (s *snapshot) cloneMetadata(without map[span.URI]struct{}) *snapshot {
	result := &snapshot{
		id:       s.id + 1,
		packages: s.packages,
		ids:      make(map[span.URI][]packageID),
		metadata: make(map[packageID]*metadata),
	}
	withoutIDs := make(map[packageID]struct{})
	for k, ids := range s.ids {
		if _, ok := without[k]; ok {
			for _, id := range ids {
				withoutIDs[id] = struct{}{}
			}
			continue
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

func (v *view) reverseDependencies(ctx context.Context, uri span.URI) map[span.URI]struct{} {
	seen := make(map[packageID]struct{})
	uris := make(map[span.URI]struct{})

	v.snapshotMu.Lock()
	defer v.snapshotMu.Unlock()

	for _, id := range v.snapshot.ids[uri] {
		v.rdeps(id, seen, uris, id)
	}
	return uris
}

func (v *view) rdeps(topID packageID, seen map[packageID]struct{}, results map[span.URI]struct{}, id packageID) {
	if _, ok := seen[id]; ok {
		return
	}
	seen[id] = struct{}{}
	m, ok := v.snapshot.metadata[id]
	if !ok {
		return
	}
	if id != topID {
		for _, uri := range m.files {
			results[uri] = struct{}{}
		}
	}
	for parentID := range m.parents {
		v.rdeps(topID, seen, results, parentID)
	}
}

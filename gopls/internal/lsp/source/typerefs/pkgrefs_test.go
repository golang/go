// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typerefs_test

import (
	"bytes"
	"context"
	"flag"
	"fmt"
	"go/token"
	"go/types"
	"os"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"golang.org/x/tools/go/gcexportdata"
	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/gopls/internal/astutil"
	"golang.org/x/tools/gopls/internal/lsp/cache"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/gopls/internal/lsp/source/typerefs"
	"golang.org/x/tools/gopls/internal/span"
	"golang.org/x/tools/internal/packagesinternal"
	"golang.org/x/tools/internal/testenv"
)

var (
	dir    = flag.String("dir", "", "dir to run go/packages from")
	query  = flag.String("query", "std", "go/packages load query to use for walkdecl tests")
	verify = flag.Bool("verify", true, "whether to verify reachable packages using export data (may be slow on large graphs)")
)

type (
	packageName    = source.PackageName
	PackageID      = source.PackageID
	ImportPath     = source.ImportPath
	PackagePath    = source.PackagePath
	Metadata       = source.Metadata
	MetadataSource = source.MetadataSource
	ParsedGoFile   = source.ParsedGoFile
)

// TestBuildPackageGraph tests the BuildPackageGraph constructor, which uses
// the reference analysis of the Refs function to build a graph of
// relationships between packages.
//
// It simulates the operation of gopls at startup: packages are loaded via
// go/packages, and their syntax+metadata analyzed to determine which packages
// are reachable from others.
//
// The test then verifies that the 'load' graph (the graph of relationships in
// export data) is a subgraph of the 'reach' graph constructed by
// BuildPackageGraph. While doing so, it constructs some statistics about the
// relative sizes of these graphs, along with the 'transitive imports' graph,
// to report the effectiveness of the reachability analysis.
//
// The following flags affect this test:
//   - dir sets the dir from which to run go/packages
//   - query sets the go/packages query to load
//   - verify toggles the verification w.r.t. the load graph (which may be
//     prohibitively expensive with large queries).
func TestBuildPackageGraph(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping with -short: loading the packages can take a long time with a cold cache")
	}
	testenv.NeedsGoBuild(t) // for go/packages

	t0 := time.Now()
	exports, meta, err := load(*query, *verify)
	if err != nil {
		t.Fatalf("loading failed: %v", err)
	}
	t.Logf("loaded %d packages in %v", len(exports), time.Since(t0))

	ctx := context.Background()
	var ids []PackageID
	for id := range exports {
		ids = append(ids, id)
	}
	sort.Slice(ids, func(i, j int) bool {
		return ids[i] < ids[j]
	})

	t0 = time.Now()
	g, err := typerefs.BuildPackageGraph(ctx, meta, ids, newParser().parse)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("building package graph took %v", time.Since(t0))

	// Collect information about the edges between packages for later analysis.
	//
	// We compare the following package graphs:
	//  - the imports graph: edges are transitive imports
	//  - the reaches graph: edges are reachability relationships through syntax
	//    of imports (as defined in the package doc)
	//  - the loads graph: edges are packages loaded through the export data of
	//    imports
	//
	// By definition, loads < reaches < imports.
	type edgeSet map[PackageID]map[PackageID]bool
	var (
		imports    = make(edgeSet) // A imports B transitively
		importedBy = make(edgeSet) // A is imported by B transitively
		reaches    = make(edgeSet) // A reaches B through top-level declaration syntax
		reachedBy  = make(edgeSet) // A is reached by B through top-level declaration syntax
		loads      = make(edgeSet) // A loads B through export data of its direct dependencies
		loadedBy   = make(edgeSet) // A is loaded by B through export data of B's direct dependencies
	)
	recordEdge := func(from, to PackageID, fwd, rev edgeSet) {
		if fwd[from] == nil {
			fwd[from] = make(map[PackageID]bool)
		}
		fwd[from][to] = true
		if rev[to] == nil {
			rev[to] = make(map[PackageID]bool)
		}
		rev[to][from] = true
	}

	exportedPackages := make(map[PackageID]*types.Package)
	importPackage := func(id PackageID) *types.Package {
		exportFile := exports[id]
		if exportFile == "" {
			return nil // no exported symbols
		}
		m := meta.Metadata(id)
		tpkg, ok := exportedPackages[id]
		if !ok {
			pkgPath := string(m.PkgPath)
			tpkg, err = importFromExportData(pkgPath, exportFile)
			if err != nil {
				t.Fatalf("importFromExportData(%s, %s) failed: %v", pkgPath, exportFile, err)
			}
			exportedPackages[id] = tpkg
		}
		return tpkg
	}

	for _, id := range ids {
		pkg, err := g.Package(ctx, id)
		if err != nil {
			t.Fatal(err)
		}
		pkg.ReachesByDeps.Elems(func(id2 PackageID) {
			recordEdge(id, id2, reaches, reachedBy)
		})

		importMap := importMap(id, meta)
		for _, id2 := range importMap {
			recordEdge(id, id2, imports, importedBy)
		}

		if *verify {
			for _, depID := range meta.Metadata(id).DepsByPkgPath {
				tpkg := importPackage(depID)
				if tpkg == nil {
					continue
				}
				for _, imp := range tpkg.Imports() {
					depID, ok := importMap[PackagePath(imp.Path())]
					if !ok {
						t.Errorf("import map (len: %d) for %s missing imported types.Package %s", len(importMap), id, imp.Path())
						continue
					}
					recordEdge(id, depID, loads, loadedBy)
				}
			}

			for depID := range loads[id] {
				if !pkg.ReachesByDeps.Contains(depID) {
					t.Errorf("package %s was imported by %s, but not detected as reachable", depID, id)
				}
			}
		}
	}

	if testing.Verbose() {
		fmt.Printf("%-52s%8s%8s%8s%8s%8s%8s\n", "package ID", "imp", "impBy", "reach", "reachBy", "load", "loadBy")
		for _, id := range ids {
			fmt.Printf("%-52s%8d%8d%8d%8d%8d%8d\n", id, len(imports[id]), len(importedBy[id]), len(reaches[id]), len(reachedBy[id]), len(loads[id]), len(loadedBy[id]))
		}
		fmt.Println(strings.Repeat("-", 100))
		fmt.Printf("%-52s%8s%8s%8s%8s%8s%8s\n", "package ID", "imp", "impBy", "reach", "reachBy", "load", "loadBy")

		avg := func(m edgeSet) float64 {
			var avg float64
			for _, id := range ids {
				s := m[id]
				avg += float64(len(s)) / float64(len(ids))
			}
			return avg
		}
		fmt.Printf("%52s%8.1f%8.1f%8.1f%8.1f%8.1f%8.1f\n", "averages:", avg(imports), avg(importedBy), avg(reaches), avg(reachedBy), avg(loads), avg(loadedBy))
	}
}

func importMap(id PackageID, meta MetadataSource) map[PackagePath]PackageID {
	imports := make(map[PackagePath]PackageID)
	var recordIDs func(PackageID)
	recordIDs = func(id PackageID) {
		m := meta.Metadata(id)
		if _, ok := imports[m.PkgPath]; ok {
			return
		}
		imports[m.PkgPath] = id
		for _, id := range m.DepsByPkgPath {
			recordIDs(id)
		}
	}
	for _, id := range meta.Metadata(id).DepsByPkgPath {
		recordIDs(id)
	}
	return imports
}

func importFromExportData(pkgPath, exportFile string) (*types.Package, error) {
	file, err := os.Open(exportFile)
	if err != nil {
		return nil, err
	}
	r, err := gcexportdata.NewReader(file)
	if err != nil {
		file.Close()
		return nil, err
	}
	fset := token.NewFileSet()
	tpkg, err := gcexportdata.Read(r, fset, make(map[string]*types.Package), pkgPath)
	file.Close()
	if err != nil {
		return nil, err
	}
	// The export file reported by go/packages is produced by the compiler, which
	// has additional package dependencies due to inlining.
	//
	// Export and re-import so that we only observe dependencies from the
	// exported API.
	var out bytes.Buffer
	err = gcexportdata.Write(&out, fset, tpkg)
	if err != nil {
		return nil, err
	}
	return gcexportdata.Read(&out, token.NewFileSet(), make(map[string]*types.Package), pkgPath)
}

func BenchmarkBuildPackageGraph(b *testing.B) {
	t0 := time.Now()
	exports, meta, err := load(*query, *verify)
	if err != nil {
		b.Fatalf("loading failed: %v", err)
	}
	b.Logf("loaded %d packages in %v", len(exports), time.Since(t0))
	ctx := context.Background()
	var ids []PackageID
	for id := range exports {
		ids = append(ids, id)
	}
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, err := typerefs.BuildPackageGraph(ctx, meta, ids, newParser().parse)
		if err != nil {
			b.Fatal(err)
		}
	}
}

type memoizedParser struct {
	mu    sync.Mutex
	files map[span.URI]*futureParse
}

type futureParse struct {
	done chan struct{}
	pgf  *ParsedGoFile
	err  error
}

func newParser() *memoizedParser {
	return &memoizedParser{
		files: make(map[span.URI]*futureParse),
	}
}

func (p *memoizedParser) parse(ctx context.Context, uri span.URI) (*ParsedGoFile, error) {
	doParse := func(ctx context.Context, uri span.URI) (*ParsedGoFile, error) {
		// TODO(adonovan): hoist this operation outside the benchmark critsec.
		content, err := os.ReadFile(uri.Filename())
		if err != nil {
			return nil, err
		}
		content = astutil.PurgeFuncBodies(content)
		pgf, _ := cache.ParseGoSrc(ctx, token.NewFileSet(), uri, content, source.ParseFull)
		return pgf, nil
	}

	p.mu.Lock()
	fut, ok := p.files[uri]
	if ok {
		p.mu.Unlock()
		select {
		case <-fut.done:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	} else {
		fut = &futureParse{done: make(chan struct{})}
		p.files[uri] = fut
		p.mu.Unlock()
		fut.pgf, fut.err = doParse(ctx, uri)
		close(fut.done)
	}
	return fut.pgf, fut.err
}

type mapMetadataSource struct {
	m map[PackageID]*Metadata
}

func (s mapMetadataSource) Metadata(id PackageID) *Metadata {
	return s.m[id]
}

// This function is a compressed version of snapshot.load from the
// internal/lsp/cache package, for use in testing.
//
// TODO(rfindley): it may be valuable to extract this logic from the snapshot,
// since it is otherwise standalone.
func load(query string, needExport bool) (map[PackageID]string, MetadataSource, error) {
	cfg := &packages.Config{
		Dir: *dir,
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
		Tests: true,
	}
	if needExport {
		cfg.Mode |= packages.NeedExportFile // ExportFile is not requested by gopls: this is used to verify reachability
	}
	pkgs, err := packages.Load(cfg, query)
	if err != nil {
		return nil, nil, err
	}

	meta := make(map[PackageID]*Metadata)
	var buildMetadata func(pkg *packages.Package)
	buildMetadata = func(pkg *packages.Package) {
		id := PackageID(pkg.ID)
		if meta[id] != nil {
			return
		}
		m := &Metadata{
			ID:         id,
			PkgPath:    PackagePath(pkg.PkgPath),
			Name:       packageName(pkg.Name),
			ForTest:    PackagePath(packagesinternal.GetForTest(pkg)),
			TypesSizes: pkg.TypesSizes,
			LoadDir:    cfg.Dir,
			Module:     pkg.Module,
			Errors:     pkg.Errors,
			DepsErrors: packagesinternal.GetDepsErrors(pkg),
		}
		meta[id] = m

		for _, filename := range pkg.CompiledGoFiles {
			m.CompiledGoFiles = append(m.CompiledGoFiles, span.URIFromPath(filename))
		}
		for _, filename := range pkg.GoFiles {
			m.GoFiles = append(m.GoFiles, span.URIFromPath(filename))
		}

		m.DepsByImpPath = make(map[ImportPath]PackageID)
		m.DepsByPkgPath = make(map[PackagePath]PackageID)
		for importPath, imported := range pkg.Imports {
			importPath := ImportPath(importPath)

			// see note in gopls/internal/lsp/cache/load.go for an explanation of this check.
			if importPath != "unsafe" && len(imported.CompiledGoFiles) == 0 {
				m.DepsByImpPath[importPath] = "" // missing
				continue
			}

			m.DepsByImpPath[importPath] = PackageID(imported.ID)
			m.DepsByPkgPath[PackagePath(imported.PkgPath)] = PackageID(imported.ID)
			buildMetadata(imported)
		}
	}

	exportFiles := make(map[PackageID]string)
	for _, pkg := range pkgs {
		exportFiles[PackageID(pkg.ID)] = pkg.ExportFile
		buildMetadata(pkg)
	}
	return exportFiles, &mapMetadataSource{meta}, nil
}

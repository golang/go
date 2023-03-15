// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"context"
	"crypto/sha256"
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"log"
	"regexp"
	"runtime"
	"sort"
	"strings"
	"sync"

	"golang.org/x/mod/module"
	"golang.org/x/sync/errgroup"
	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/gopls/internal/lsp/filecache"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/gopls/internal/lsp/source/methodsets"
	"golang.org/x/tools/gopls/internal/lsp/source/xrefs"
	"golang.org/x/tools/gopls/internal/span"
	"golang.org/x/tools/internal/bug"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/event/tag"
	"golang.org/x/tools/internal/gcimporter"
	"golang.org/x/tools/internal/memoize"
	"golang.org/x/tools/internal/packagesinternal"
	"golang.org/x/tools/internal/typeparams"
	"golang.org/x/tools/internal/typesinternal"
)

// A typeCheckBatch holds data for a logical type-checking operation, which may
// type-check many unrelated packages.
//
// It shares state such as parsed files and imports, to optimize type-checking
// for packages with overlapping dependency graphs.
type typeCheckBatch struct {
	meta *metadataGraph

	cpulimit    chan struct{}                     // concurrency limiter for CPU-bound operations
	needSyntax  map[PackageID]bool                // packages that need type-checked syntax
	parsedFiles map[span.URI]*source.ParsedGoFile // parsed files necessary for type-checking
	fset        *token.FileSet                    // FileSet describing all parsed files

	// Promises holds promises to either read export data for the package, or
	// parse and type-check its syntax.
	//
	// The return value of these promises is not used: after promises are
	// awaited, they must write an entry into the imports map.
	promises map[PackageID]*memoize.Promise

	mu         sync.Mutex
	needFiles  map[span.URI]source.FileHandle // de-duplicated file handles required for type-checking
	imports    map[PackageID]pkgOrErr         // types.Packages to use for importing
	exportData map[PackageID][]byte
	packages   map[PackageID]*Package
}

type pkgOrErr struct {
	pkg *types.Package
	err error
}

// TypeCheck type-checks the specified packages.
//
// The resulting packages slice always contains len(ids) entries, though some
// of them may be nil if (and only if) the resulting error is non-nil.
//
// An error is returned if any of the requested packages fail to type-check.
// This is different from having type-checking errors: a failure to type-check
// indicates context cancellation or otherwise significant failure to perform
// the type-checking operation.
func (s *snapshot) TypeCheck(ctx context.Context, ids ...PackageID) ([]source.Package, error) {
	// Build up shared state for efficient type-checking.
	b := &typeCheckBatch{
		cpulimit:    make(chan struct{}, runtime.GOMAXPROCS(0)),
		needSyntax:  make(map[PackageID]bool),
		parsedFiles: make(map[span.URI]*source.ParsedGoFile),
		// fset is built during the parsing pass.

		needFiles:  make(map[span.URI]source.FileHandle),
		promises:   make(map[PackageID]*memoize.Promise),
		imports:    make(map[PackageID]pkgOrErr),
		exportData: make(map[PackageID][]byte),
		packages:   make(map[PackageID]*Package),
	}

	// Check for existing active packages.
	//
	// Since gopls can't depend on package identity, any instance of the
	// requested package must be ok to return.
	//
	// This is an optimization to avoid redundant type-checking: following
	// changes to an open package many LSP clients send several successive
	// requests for package information for the modified package (semantic
	// tokens, code lens, inlay hints, etc.)
	pkgs := make([]source.Package, len(ids))
	for i, id := range ids {
		if pkg := s.getActivePackage(id); pkg != nil {
			pkgs[i] = pkg
		} else {
			b.needSyntax[id] = true
		}
	}

	if len(b.needSyntax) == 0 {
		return pkgs, nil
	}

	// Capture metadata once to ensure a consistent view.
	s.mu.Lock()
	b.meta = s.meta
	s.mu.Unlock()

	//  -- Step 1: assemble the promises graph --
	var (
		needExportData = make(map[PackageID]packageHandleKey)
		packageHandles = make(map[PackageID]*packageHandle)
	)

	// collectPromises collects promises to load packages from export data or
	// type-check.
	var collectPromises func(PackageID) error
	collectPromises = func(id PackageID) error {
		if _, ok := b.promises[id]; ok {
			return nil
		}
		b.promises[id] = nil // break cycles

		m := b.meta.metadata[id]
		if m == nil {
			return bug.Errorf("missing metadata for %v", id)
		}
		for _, id := range m.DepsByPkgPath {
			if err := collectPromises(id); err != nil {
				return err
			}
		}

		// Note that we can't reuse active packages here, as they will have the
		// wrong FileSet. Any active packages that exist as dependencies of other
		// packages will need to be loaded from export data.
		ph, err := s.buildPackageHandle(ctx, id)
		if err != nil {
			return err
		}
		packageHandles[id] = ph

		if b.needSyntax[id] {
			// We will need to parse and type-check this package.
			//
			// We may also need to parse and type-check if export data is missing,
			// but that is handled after fetching export data below.
			b.addNeededFiles(ph)
		} else if id != "unsafe" { // we can't load export data for unsafe
			needExportData[id] = ph.key
		}

		debugName := fmt.Sprintf("check(%s)", id)
		b.promises[id] = memoize.NewPromise(debugName, func(ctx context.Context, _ interface{}) interface{} {
			pkg, err := b.processPackage(ctx, ph)
			b.mu.Lock()
			b.imports[m.ID] = pkgOrErr{pkg, err}
			b.mu.Unlock()
			return nil
		})
		return nil
	}
	for id := range b.needSyntax {
		collectPromises(id)
	}

	// -- Step 2: collect export data --
	//
	// This must be done before parsing in order to determine which files must be
	// parsed.
	{
		var g errgroup.Group
		for id, key := range needExportData {
			id := id
			key := key
			g.Go(func() error {
				data, err := filecache.Get(exportDataKind, key)
				if err != nil {
					if err == filecache.ErrNotFound {
						ph := packageHandles[id]
						b.addNeededFiles(ph) // we will need to parse and type check
						return nil           // ok: we will type check later
					}
					return err
				}
				b.mu.Lock()
				b.exportData[id] = data
				b.mu.Unlock()
				return nil
			})
		}
		if err := g.Wait(); err != nil {
			return pkgs, err
		}
	}

	// -- Step 3: parse files required for type checking. --
	//
	// Parse all necessary files in parallel. Unfortunately we can't start
	// parsing each package's file as soon as we discover that it is a syntax
	// package, because the parseCache cannot add files to an existing FileSet.
	{
		var fhs []source.FileHandle
		for _, fh := range b.needFiles {
			fhs = append(fhs, fh)
		}
		pgfs, fset, err := s.parseCache.parseFiles(ctx, source.ParseFull, fhs...)
		if err != nil {
			return pkgs, err
		}
		for _, pgf := range pgfs {
			b.parsedFiles[pgf.URI] = pgf
		}
		b.fset = fset
	}

	// -- Step 4: await results --
	//
	// Start a single goroutine for each promise.
	{
		var g errgroup.Group
		// TODO(rfindley): find a good way to limit concurrency of type-checking,
		// which is CPU bound at this point.
		//
		// (calling g.SetLimit here is mostly ineffective, as promises are
		// recursively concurrent.)
		for _, promise := range b.promises {
			promise := promise
			g.Go(func() error {
				_, err := promise.Get(ctx, nil)
				return err
			})
		}
		if err := g.Wait(); err != nil {
			return pkgs, err
		}
	}

	// Fill in the gaps of the results slice.
	var firstErr error
	for i, id := range ids {
		if pkgs[i] != nil {
			continue
		}
		if err := b.imports[id].err; err != nil {
			if firstErr == nil {
				firstErr = err
			}
			continue
		}
		pkg := b.packages[id]
		if pkg == nil {
			panic("nil package")
		}
		if alt := s.memoizeActivePackage(id, pkg); alt != nil && alt != pkg {
			// pkg is an open package, but we've lost a race and an existing package
			// has already been memoized.
			pkg = alt
		}
		pkgs[i] = pkg
	}

	return pkgs, firstErr
}

// addNeededFiles records the files necessary for type-checking ph, for later
// parsing.
func (b *typeCheckBatch) addNeededFiles(ph *packageHandle) {
	b.mu.Lock()
	defer b.mu.Unlock()

	// Technically for export-only packages we only need compiledGoFiles, but
	// these slices are usually redundant.
	for _, fh := range ph.inputs.goFiles {
		b.needFiles[fh.URI()] = fh
	}
	for _, fh := range ph.inputs.compiledGoFiles {
		b.needFiles[fh.URI()] = fh
	}
}

// processPackage processes the package handle for the type checking batch,
// which may involve any one of importing, type-checking for import, or
// type-checking for syntax, depending on the requested syntax packages and
// available export data.
func (b *typeCheckBatch) processPackage(ctx context.Context, ph *packageHandle) (*types.Package, error) {
	if err := b.awaitPredecessors(ctx, ph.m); err != nil {
		return nil, err
	}

	// Wait to acquire CPU token.
	//
	// Note: it is important to acquire this token only after awaiting
	// predecessors, to avoid a starvation lock.
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case b.cpulimit <- struct{}{}:
		defer func() {
			<-b.cpulimit // release CPU token
		}()
	}

	b.mu.Lock()
	data, ok := b.exportData[ph.m.ID]
	b.mu.Unlock()

	if ok {
		// We need export data, and have it.
		return b.importPackage(ctx, ph.m, data)
	}

	if !b.needSyntax[ph.m.ID] {
		// We need only a types.Package, but don't have export data.
		// Type-check as fast as possible (skipping function bodies).
		return b.checkPackageForImport(ctx, ph)
	}

	// We need a syntax package.
	syntaxPkg, err := b.checkPackage(ctx, ph)
	if err != nil {
		return nil, err
	}

	b.mu.Lock()
	b.packages[ph.m.ID] = syntaxPkg
	b.mu.Unlock()
	return syntaxPkg.pkg.types, nil
}

// importPackage loads the given package from its export data in p.exportData
// (which must already be populated).
func (b *typeCheckBatch) importPackage(ctx context.Context, m *source.Metadata, data []byte) (*types.Package, error) {
	impMap, errMap := b.importMap(m.ID)
	// Any failure to populate an import will cause confusing errors from
	// IImportShallow below.
	for path, err := range errMap {
		return nil, fmt.Errorf("error importing %q for %q: %v", path, m.ID, err)
	}

	// TODO(rfindley): collect "deep" hashes here using the provided
	// callback, for precise pruning.
	imported, err := gcimporter.IImportShallow(b.fset, gcimporter.GetPackageFromMap(impMap), data, string(m.PkgPath), func(*types.Package, string) {})
	if err != nil {
		return nil, bug.Errorf("invalid export data for %q: %v", m.ID, err)
	}
	return imported, nil
}

// checkPackageForImport type checks, but skips function bodies and does not
// record syntax information.
func (b *typeCheckBatch) checkPackageForImport(ctx context.Context, ph *packageHandle) (*types.Package, error) {
	if ph.m.ID == "unsafe" {
		return types.Unsafe, nil
	}
	impMap, errMap := b.importMap(ph.inputs.id)
	onError := func(e error) {
		// Ignore errors for exporting.
	}
	cfg := b.typesConfig(ph.inputs, onError, impMap, errMap)
	var files []*ast.File
	for _, fh := range ph.inputs.compiledGoFiles {
		pgf := b.parsedFiles[fh.URI()]
		if pgf == nil {
			return nil, fmt.Errorf("compiled go file %q failed to parse", fh.URI().Filename())
		}
		files = append(files, pgf.File)
	}
	cfg.IgnoreFuncBodies = true
	pkg := types.NewPackage(string(ph.inputs.pkgPath), string(ph.inputs.name))
	check := types.NewChecker(cfg, b.fset, pkg, nil)

	_ = check.Files(files) // ignore errors

	// If the context was cancelled, we may have returned a ton of transient
	// errors to the type checker. Swallow them.
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}

	// Asynchronously record export data.
	go func() {
		exportData, err := gcimporter.IExportShallow(b.fset, pkg)
		if err != nil {
			bug.Reportf("exporting package %v: %v", ph.m.ID, err)
			return
		}
		if err := filecache.Set(exportDataKind, ph.key, exportData); err != nil {
			event.Error(ctx, fmt.Sprintf("storing export data for %s", ph.m.ID), err)
		}
	}()
	return pkg, nil
}

// checkPackage "fully type checks" to produce a syntax package.
func (b *typeCheckBatch) checkPackage(ctx context.Context, ph *packageHandle) (*Package, error) {
	// TODO(rfindley): refactor to inline typeCheckImpl here. There is no need
	// for so many layers to build up the package
	// (checkPackage->typeCheckImpl->doTypeCheck).
	pkg, err := typeCheckImpl(ctx, b, ph.inputs)

	if err == nil {
		// Write package data to disk asynchronously.
		go func() {
			toCache := map[string][]byte{
				xrefsKind:       pkg.xrefs,
				methodSetsKind:  pkg.methodsets.Encode(),
				diagnosticsKind: encodeDiagnostics(pkg.diagnostics),
			}

			if ph.m.ID != "unsafe" { // unsafe cannot be exported
				exportData, err := gcimporter.IExportShallow(pkg.fset, pkg.types)
				if err != nil {
					bug.Reportf("exporting package %v: %v", ph.m.ID, err)
				} else {
					toCache[exportDataKind] = exportData
				}
			}

			for kind, data := range toCache {
				if err := filecache.Set(kind, ph.key, data); err != nil {
					event.Error(ctx, fmt.Sprintf("storing %s data for %s", kind, ph.m.ID), err)
				}
			}
		}()
	}

	return &Package{ph.m, pkg}, err
}

// awaitPredecessors awaits all promises for m.DepsByPkgPath, returning an
// error if awaiting failed due to context cancellation or if there was an
// unrecoverable error loading export data.
func (b *typeCheckBatch) awaitPredecessors(ctx context.Context, m *source.Metadata) error {
	for _, depID := range m.DepsByPkgPath {
		depID := depID
		if p, ok := b.promises[depID]; ok {
			if _, err := p.Get(ctx, nil); err != nil {
				return err
			}
		}
	}
	return nil
}

// importMap returns an import map for the given package ID, populated with
// type-checked packages for its dependencies. It is intended for compatibility
// with gcimporter.IImportShallow, so the first result uses the map signature
// of that API, where keys are package path strings.
//
// importMap must only be used once all promises for dependencies of id have
// been awaited.
//
// For any missing packages, importMap returns an entry in the resulting errMap
// reporting the error for that package.
//
// Invariant: for all recursive dependencies, either impMap[path] or
// errMap[path] is set.
func (b *typeCheckBatch) importMap(id PackageID) (impMap map[string]*types.Package, errMap map[PackagePath]error) {
	impMap = make(map[string]*types.Package)
	outerID := id
	var populateDepsOf func(m *source.Metadata)
	populateDepsOf = func(parent *source.Metadata) {
		for _, id := range parent.DepsByPkgPath {
			m := b.meta.metadata[id]
			if _, ok := impMap[string(m.PkgPath)]; ok {
				continue
			}
			if _, ok := errMap[m.PkgPath]; ok {
				continue
			}
			b.mu.Lock()
			result, ok := b.imports[m.ID]
			b.mu.Unlock()
			if !ok {
				panic(fmt.Sprintf("import map for %q missing package data for %q", outerID, m.ID))
			}
			// We may fail to produce a package due to e.g. context cancellation
			// (handled elsewhere), or some catastrophic failure such as a package with
			// no files.
			switch {
			case result.err != nil:
				if errMap == nil {
					errMap = make(map[PackagePath]error)
				}
				errMap[m.PkgPath] = result.err
			case result.pkg != nil:
				impMap[string(m.PkgPath)] = result.pkg
			default:
				panic("invalid import for " + id)
			}
			populateDepsOf(m)
		}
	}
	m := b.meta.metadata[id]
	populateDepsOf(m)
	return impMap, errMap
}

// packageData holds binary data (e.g. types, xrefs) extracted from a syntax
// package.
type packageData struct {
	m    *source.Metadata
	data []byte
}

// getPackageData gets package data (e.g. types, xrefs) for the requested ids,
// either loading from the file-based cache or type-checking and extracting
// data using the provided get function.
func (s *snapshot) getPackageData(ctx context.Context, kind string, ids []PackageID, get func(*syntaxPackage) []byte) ([]*packageData, error) {
	needIDs := make([]PackageID, len(ids))
	pkgData := make([]*packageData, len(ids))

	// Compute package keys and query file cache.
	var grp errgroup.Group
	for i, id := range ids {
		i, id := i, id
		grp.Go(func() error {
			ph, err := s.buildPackageHandle(ctx, id)
			if err != nil {
				return err
			}
			data, err := filecache.Get(kind, ph.key)
			if err == nil { // hit
				pkgData[i] = &packageData{m: ph.m, data: data}
			} else if err == filecache.ErrNotFound { // miss
				needIDs[i] = id
				err = nil
			}
			return err
		})
	}
	if err := grp.Wait(); err != nil {
		return pkgData, err
	}

	// Compact needIDs (which was sparse to avoid the need for a mutex).
	out := needIDs[:0]
	for _, id := range needIDs {
		if id != "" {
			out = append(out, id)
		}
	}
	needIDs = out

	// Type-check the packages for which we got file-cache misses.
	pkgs, err := s.TypeCheck(ctx, needIDs...)
	if err != nil {
		return pkgData, err
	}

	pkgMap := make(map[PackageID]source.Package)
	for i, id := range needIDs {
		pkgMap[id] = pkgs[i]
	}

	// Fill in the gaps using data derived from type checking.
	for i, id := range ids {
		if pkgData[i] != nil {
			continue
		}
		result := pkgMap[id]
		if result == nil {
			panic(fmt.Sprintf("missing type-check result for %s", id))
		}
		data := get(result.(*Package).pkg)
		pkgData[i] = &packageData{m: result.Metadata(), data: data}
	}

	return pkgData, nil
}

type packageHandleKey source.Hash

// A packageHandle holds package information, some of which may not be fully
// evaluated.
//
// The only methods on packageHandle that are safe to call before calling await
// are Metadata and await itself.
type packageHandle struct {
	m *source.Metadata

	inputs typeCheckInputs

	// key is the hashed key for the package.
	//
	// It includes the all bits of the transitive closure of
	// dependencies's sources. This is more than type checking
	// really depends on: export data of direct deps should be
	// enough. (The key for analysis actions could similarly
	// hash only Facts of direct dependencies.)
	key packageHandleKey

	// Note: as an optimization, we could join in-flight type-checking by
	// recording a transient ref-counted promise here.
	// (This was done previously, but proved to be a premature optimization).
}

// buildPackageHandle returns a handle for the future results of
// type-checking the package identified by id in the given mode.
// It assumes that the given ID already has metadata available, so it does not
// attempt to reload missing or invalid metadata. The caller must reload
// metadata if needed.
func (s *snapshot) buildPackageHandle(ctx context.Context, id PackageID) (*packageHandle, error) {
	s.mu.Lock()
	entry, hit := s.packages.Get(id)
	m := s.meta.metadata[id]
	s.mu.Unlock()

	if m == nil {
		return nil, fmt.Errorf("no metadata for %s", id)
	}

	if hit {
		return entry.(*packageHandle), nil
	}

	inputs, err := s.typeCheckInputs(ctx, m)
	if err != nil {
		return nil, err
	}
	// All the file reading has now been done.
	// Create a handle for the result of type checking.
	phKey := computePackageKey(s, inputs)
	ph := &packageHandle{
		m:      m,
		inputs: inputs,
		key:    phKey,
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	// Check that the metadata has not changed
	// (which should invalidate this handle).
	//
	// (In future, handles should form a graph with edges from a
	// packageHandle to the handles for parsing its files and the
	// handles for type-checking its immediate deps, at which
	// point there will be no need to even access s.meta.)
	if s.meta.metadata[ph.m.ID] != ph.m {
		// TODO(rfindley): this should be bug.Errorf.
		return nil, fmt.Errorf("stale metadata for %s", ph.m.ID)
	}

	// Check cache again in case another goroutine got there first.
	if prev, ok := s.packages.Get(id); ok {
		prevPH := prev.(*packageHandle)
		if prevPH.m != ph.m {
			return nil, bug.Errorf("existing package handle does not match for %s", ph.m.ID)
		}
		return prevPH, nil
	}

	s.packages.Set(id, ph, nil)
	return ph, nil
}

// typeCheckInputs contains the inputs of a call to typeCheckImpl, which
// type-checks a package.
//
// Part of the purpose of this type is to keep type checking in-sync with the
// package handle key, by explicitly identifying the inputs to type checking.
type typeCheckInputs struct {
	id PackageID

	// Used for type checking:
	pkgPath                  PackagePath
	name                     PackageName
	goFiles, compiledGoFiles []source.FileHandle
	sizes                    types.Sizes
	deps                     map[PackageID]*packageHandle
	depsByImpPath            map[ImportPath]PackageID
	goVersion                string // packages.Module.GoVersion, e.g. "1.18"

	// Used for type check diagnostics:
	relatedInformation bool
	linkTarget         string
	moduleMode         bool
}

func (s *snapshot) typeCheckInputs(ctx context.Context, m *source.Metadata) (typeCheckInputs, error) {
	deps := make(map[PackageID]*packageHandle)
	for _, depID := range m.DepsByPkgPath {
		depHandle, err := s.buildPackageHandle(ctx, depID)
		if err != nil {
			// If err is non-nil, we either have an invalid dependency, or a
			// catastrophic failure to read a file (context cancellation or
			// permission issues).
			//
			// We don't want one bad dependency to prevent us from type-checking the
			// package -- we should instead get an import error. So we only abort
			// this operation if the context is cancelled.
			//
			// We could do a better job of handling permission errors on files, but
			// this is rare, and it is reasonable to treat the same an invalid
			// dependency.
			event.Error(ctx, fmt.Sprintf("%s: no dep handle for %s", m.ID, depID), err, source.SnapshotLabels(s)...)
			if ctx.Err() != nil {
				return typeCheckInputs{}, ctx.Err() // cancelled
			}
			continue
		}
		deps[depID] = depHandle
	}

	// Read both lists of files of this package.
	//
	// Parallelism is not necessary here as the files will have already been
	// pre-read at load time.
	//
	// goFiles aren't presented to the type checker--nor
	// are they included in the key, unsoundly--but their
	// syntax trees are available from (*pkg).File(URI).
	// TODO(adonovan): consider parsing them on demand?
	// The need should be rare.
	goFiles, err := readFiles(ctx, s, m.GoFiles)
	if err != nil {
		return typeCheckInputs{}, err
	}
	compiledGoFiles, err := readFiles(ctx, s, m.CompiledGoFiles)
	if err != nil {
		return typeCheckInputs{}, err
	}

	goVersion := ""
	if m.Module != nil && m.Module.GoVersion != "" {
		goVersion = m.Module.GoVersion
	}

	return typeCheckInputs{
		id:              m.ID,
		pkgPath:         m.PkgPath,
		name:            m.Name,
		goFiles:         goFiles,
		compiledGoFiles: compiledGoFiles,
		sizes:           m.TypesSizes,
		deps:            deps,
		depsByImpPath:   m.DepsByImpPath,
		goVersion:       goVersion,

		relatedInformation: s.view.Options().RelatedInformationSupported,
		linkTarget:         s.view.Options().LinkTarget,
		moduleMode:         s.moduleMode(),
	}, nil
}

// readFiles reads the content of each file URL from the source
// (e.g. snapshot or cache).
func readFiles(ctx context.Context, fs source.FileSource, uris []span.URI) (_ []source.FileHandle, err error) {
	fhs := make([]source.FileHandle, len(uris))
	for i, uri := range uris {
		fhs[i], err = fs.ReadFile(ctx, uri)
		if err != nil {
			return nil, err
		}
	}
	return fhs, nil
}

// computePackageKey returns a key representing the act of type checking
// a package named id containing the specified files, metadata, and
// combined dependency hash.
func computePackageKey(s *snapshot, inputs typeCheckInputs) packageHandleKey {
	hasher := sha256.New()

	// In principle, a key must be the hash of an
	// unambiguous encoding of all the relevant data.
	// If it's ambiguous, we risk collisions.

	// package identifiers
	fmt.Fprintf(hasher, "package: %s %s %s\n", inputs.id, inputs.name, inputs.pkgPath)

	// module Go version
	fmt.Fprintf(hasher, "go %s\n", inputs.goVersion)

	// import map
	importPaths := make([]string, 0, len(inputs.depsByImpPath))
	for impPath := range inputs.depsByImpPath {
		importPaths = append(importPaths, string(impPath))
	}
	sort.Strings(importPaths)
	for _, impPath := range importPaths {
		fmt.Fprintf(hasher, "import %s %s", impPath, string(inputs.depsByImpPath[ImportPath(impPath)]))
	}

	// deps, in PackageID order
	depIDs := make([]string, 0, len(inputs.deps))
	for depID := range inputs.deps {
		depIDs = append(depIDs, string(depID))
	}
	sort.Strings(depIDs)
	for _, depID := range depIDs {
		dep := inputs.deps[PackageID(depID)]
		fmt.Fprintf(hasher, "dep: %s key:%s\n", dep.m.PkgPath, dep.key)
	}

	// file names and contents
	fmt.Fprintf(hasher, "compiledGoFiles: %d\n", len(inputs.compiledGoFiles))
	for _, fh := range inputs.compiledGoFiles {
		fmt.Fprintln(hasher, fh.FileIdentity())
	}
	fmt.Fprintf(hasher, "goFiles: %d\n", len(inputs.goFiles))
	for _, fh := range inputs.goFiles {
		fmt.Fprintln(hasher, fh.FileIdentity())
	}

	// types sizes
	sz := inputs.sizes.(*types.StdSizes)
	fmt.Fprintf(hasher, "sizes: %d %d\n", sz.WordSize, sz.MaxAlign)

	fmt.Fprintf(hasher, "relatedInformation: %t\n", inputs.relatedInformation)
	fmt.Fprintf(hasher, "linkTarget: %s\n", inputs.linkTarget)
	fmt.Fprintf(hasher, "moduleMode: %t\n", inputs.moduleMode)

	var hash [sha256.Size]byte
	hasher.Sum(hash[:0])
	return packageHandleKey(hash)
}

// typeCheckImpl type checks the parsed source files in compiledGoFiles.
// (The resulting pkg also holds the parsed but not type-checked goFiles.)
// deps holds the future results of type-checking the direct dependencies.
func typeCheckImpl(ctx context.Context, b *typeCheckBatch, inputs typeCheckInputs) (*syntaxPackage, error) {
	ctx, done := event.Start(ctx, "cache.typeCheck", tag.Package.Of(string(inputs.id)))
	defer done()

	pkg, err := doTypeCheck(ctx, b, inputs)
	if err != nil {
		return nil, err
	}
	pkg.methodsets = methodsets.NewIndex(pkg.fset, pkg.types)
	pkg.xrefs = xrefs.Index(pkg.compiledGoFiles, pkg.types, pkg.typesInfo)

	// Our heuristic for whether to show type checking errors is:
	//  + If any file was 'fixed', don't show type checking errors as we
	//    can't guarantee that they reference accurate locations in the source.
	//  + If there is a parse error _in the current file_, suppress type
	//    errors in that file.
	//  + Otherwise, show type errors even in the presence of parse errors in
	//    other package files. go/types attempts to suppress follow-on errors
	//    due to bad syntax, so on balance type checking errors still provide
	//    a decent signal/noise ratio as long as the file in question parses.

	// Track URIs with parse errors so that we can suppress type errors for these
	// files.
	unparseable := map[span.URI]bool{}
	for _, e := range pkg.parseErrors {
		diags, err := parseErrorDiagnostics(pkg, e)
		if err != nil {
			event.Error(ctx, "unable to compute positions for parse errors", err, tag.Package.Of(string(inputs.id)))
			continue
		}
		for _, diag := range diags {
			unparseable[diag.URI] = true
			pkg.diagnostics = append(pkg.diagnostics, diag)
		}
	}

	if pkg.hasFixedFiles {
		return pkg, nil
	}

	unexpanded := pkg.typeErrors
	pkg.typeErrors = nil
	for _, e := range expandErrors(unexpanded, inputs.relatedInformation) {
		diags, err := typeErrorDiagnostics(inputs.moduleMode, inputs.linkTarget, pkg, e)
		if err != nil {
			// If we fail here and there are no parse errors, it means we are hiding
			// a valid type-checking error from the user. This must be a bug.
			if len(pkg.parseErrors) == 0 {
				bug.Reportf("failed to compute position for type error %v: %v", e, err)
			}
			continue
		}
		pkg.typeErrors = append(pkg.typeErrors, e.primary)
		for _, diag := range diags {
			// If the file didn't parse cleanly, it is highly likely that type
			// checking errors will be confusing or redundant. But otherwise, type
			// checking usually provides a good enough signal to include.
			if !unparseable[diag.URI] {
				pkg.diagnostics = append(pkg.diagnostics, diag)
			}
		}
	}

	return pkg, nil
}

var goVersionRx = regexp.MustCompile(`^go([1-9][0-9]*)\.(0|[1-9][0-9]*)$`)

func doTypeCheck(ctx context.Context, b *typeCheckBatch, inputs typeCheckInputs) (*syntaxPackage, error) {
	impMap, errMap := b.importMap(inputs.id)
	pkg := &syntaxPackage{
		id:    inputs.id,
		fset:  b.fset, // must match parse call below
		types: types.NewPackage(string(inputs.pkgPath), string(inputs.name)),
		typesInfo: &types.Info{
			Types:      make(map[ast.Expr]types.TypeAndValue),
			Defs:       make(map[*ast.Ident]types.Object),
			Uses:       make(map[*ast.Ident]types.Object),
			Implicits:  make(map[ast.Node]types.Object),
			Selections: make(map[*ast.SelectorExpr]*types.Selection),
			Scopes:     make(map[ast.Node]*types.Scope),
		},
		importMap: impMap,
	}
	typeparams.InitInstanceInfo(pkg.typesInfo)

	// Collect parsed files from the type check pass, capturing parse errors from
	// compiled files.
	for _, fh := range inputs.goFiles {
		pgf := b.parsedFiles[fh.URI()]
		if pgf == nil {
			// If go/packages told us that a file is in a package, it should be
			// parseable (after all, it was parsed by go list).
			return nil, bug.Errorf("go file %q failed to parse", fh.URI().Filename())
		}
		pkg.goFiles = append(pkg.goFiles, pgf)
	}
	for _, fh := range inputs.compiledGoFiles {
		pgf := b.parsedFiles[fh.URI()]
		if pgf == nil {
			return nil, fmt.Errorf("compiled go file %q failed to parse", fh.URI().Filename())
		}
		if pgf.ParseErr != nil {
			pkg.parseErrors = append(pkg.parseErrors, pgf.ParseErr)
		}
		pkg.compiledGoFiles = append(pkg.compiledGoFiles, pgf)
	}

	// Use the default type information for the unsafe package.
	if inputs.pkgPath == "unsafe" {
		// Don't type check Unsafe: it's unnecessary, and doing so exposes a data
		// race to Unsafe.completed.
		pkg.types = types.Unsafe
		return pkg, nil
	}

	if len(pkg.compiledGoFiles) == 0 {
		// No files most likely means go/packages failed.
		//
		// TODO(rfindley): in the past, we would capture go list errors in this
		// case, to present go list errors to the user. However we had no tests for
		// this behavior. It is unclear if anything better can be done here.
		return nil, fmt.Errorf("no parsed files for package %s", inputs.pkgPath)
	}

	onError := func(e error) {
		pkg.typeErrors = append(pkg.typeErrors, e.(types.Error))
	}
	cfg := b.typesConfig(inputs, onError, impMap, errMap)

	check := types.NewChecker(cfg, pkg.fset, pkg.types, pkg.typesInfo)

	var files []*ast.File
	for _, cgf := range pkg.compiledGoFiles {
		files = append(files, cgf.File)
	}

	// Type checking errors are handled via the config, so ignore them here.
	_ = check.Files(files) // 50us-15ms, depending on size of package

	// If the context was cancelled, we may have returned a ton of transient
	// errors to the type checker. Swallow them.
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}
	return pkg, nil
}

func (b *typeCheckBatch) typesConfig(inputs typeCheckInputs, onError func(e error), impMap map[string]*types.Package, errMap map[PackagePath]error) *types.Config {
	cfg := &types.Config{
		Sizes: inputs.sizes,
		Error: onError,
		Importer: importerFunc(func(path string) (*types.Package, error) {
			// While all of the import errors could be reported
			// based on the metadata before we start type checking,
			// reporting them via types.Importer places the errors
			// at the correct source location.
			id, ok := inputs.depsByImpPath[ImportPath(path)]
			if !ok {
				// If the import declaration is broken,
				// go list may fail to report metadata about it.
				// See TestFixImportDecl for an example.
				return nil, fmt.Errorf("missing metadata for import of %q", path)
			}
			depPH := inputs.deps[id]
			if depPH == nil {
				// e.g. missing metadata for dependencies in buildPackageHandle
				return nil, missingPkgError(path, inputs.moduleMode)
			}
			if !source.IsValidImport(inputs.pkgPath, depPH.m.PkgPath) {
				return nil, fmt.Errorf("invalid use of internal package %q", path)
			}
			pkg, ok := impMap[string(depPH.m.PkgPath)]
			if !ok {
				err := errMap[depPH.m.PkgPath]
				if err == nil {
					log.Fatalf("neither pkg nor error is set")
				}
				return nil, err
			}
			return pkg, nil
		}),
	}

	if inputs.goVersion != "" {
		goVersion := "go" + inputs.goVersion
		// types.NewChecker panics if GoVersion is invalid. An unparsable mod
		// file should probably stop us before we get here, but double check
		// just in case.
		if goVersionRx.MatchString(goVersion) {
			typesinternal.SetGoVersion(cfg, goVersion)
		}
	}

	// We want to type check cgo code if go/types supports it.
	// We passed typecheckCgo to go/packages when we Loaded.
	typesinternal.SetUsesCgo(cfg)
	return cfg
}

// depsErrors creates diagnostics for each metadata error (e.g. import cycle).
// These may be attached to import declarations in the transitive source files
// of pkg, or to 'requires' declarations in the package's go.mod file.
//
// TODO(rfindley): move this to load.go
func depsErrors(ctx context.Context, m *source.Metadata, meta *metadataGraph, fs source.FileSource, workspacePackages map[PackageID]PackagePath) ([]*source.Diagnostic, error) {
	// Select packages that can't be found, and were imported in non-workspace packages.
	// Workspace packages already show their own errors.
	var relevantErrors []*packagesinternal.PackageError
	for _, depsError := range m.DepsErrors {
		// Up to Go 1.15, the missing package was included in the stack, which
		// was presumably a bug. We want the next one up.
		directImporterIdx := len(depsError.ImportStack) - 1
		if directImporterIdx < 0 {
			continue
		}

		directImporter := depsError.ImportStack[directImporterIdx]
		if _, ok := workspacePackages[PackageID(directImporter)]; ok {
			continue
		}
		relevantErrors = append(relevantErrors, depsError)
	}

	// Don't build the import index for nothing.
	if len(relevantErrors) == 0 {
		return nil, nil
	}

	// Subsequent checks require Go files.
	if len(m.CompiledGoFiles) == 0 {
		return nil, nil
	}

	// Build an index of all imports in the package.
	type fileImport struct {
		cgf *source.ParsedGoFile
		imp *ast.ImportSpec
	}
	allImports := map[string][]fileImport{}
	for _, uri := range m.CompiledGoFiles {
		pgf, err := parseGoURI(ctx, fs, uri, source.ParseHeader)
		if err != nil {
			return nil, err
		}
		fset := source.FileSetFor(pgf.Tok)
		// TODO(adonovan): modify Imports() to accept a single token.File (cgf.Tok).
		for _, group := range astutil.Imports(fset, pgf.File) {
			for _, imp := range group {
				if imp.Path == nil {
					continue
				}
				path := strings.Trim(imp.Path.Value, `"`)
				allImports[path] = append(allImports[path], fileImport{pgf, imp})
			}
		}
	}

	// Apply a diagnostic to any import involved in the error, stopping once
	// we reach the workspace.
	var errors []*source.Diagnostic
	for _, depErr := range relevantErrors {
		for i := len(depErr.ImportStack) - 1; i >= 0; i-- {
			item := depErr.ImportStack[i]
			if _, ok := workspacePackages[PackageID(item)]; ok {
				break
			}

			for _, imp := range allImports[item] {
				rng, err := imp.cgf.NodeRange(imp.imp)
				if err != nil {
					return nil, err
				}
				fixes, err := goGetQuickFixes(m.Module != nil, imp.cgf.URI, item)
				if err != nil {
					return nil, err
				}
				errors = append(errors, &source.Diagnostic{
					URI:            imp.cgf.URI,
					Range:          rng,
					Severity:       protocol.SeverityError,
					Source:         source.TypeError,
					Message:        fmt.Sprintf("error while importing %v: %v", item, depErr.Err),
					SuggestedFixes: fixes,
				})
			}
		}
	}

	modFile, err := nearestModFile(ctx, m.CompiledGoFiles[0], fs)
	if err != nil {
		return nil, err
	}
	pm, err := parseModURI(ctx, fs, modFile)
	if err != nil {
		return nil, err
	}

	// Add a diagnostic to the module that contained the lowest-level import of
	// the missing package.
	for _, depErr := range relevantErrors {
		for i := len(depErr.ImportStack) - 1; i >= 0; i-- {
			item := depErr.ImportStack[i]
			m := meta.metadata[PackageID(item)]
			if m == nil || m.Module == nil {
				continue
			}
			modVer := module.Version{Path: m.Module.Path, Version: m.Module.Version}
			reference := findModuleReference(pm.File, modVer)
			if reference == nil {
				continue
			}
			rng, err := pm.Mapper.OffsetRange(reference.Start.Byte, reference.End.Byte)
			if err != nil {
				return nil, err
			}
			fixes, err := goGetQuickFixes(true, pm.URI, item)
			if err != nil {
				return nil, err
			}
			errors = append(errors, &source.Diagnostic{
				URI:            pm.URI,
				Range:          rng,
				Severity:       protocol.SeverityError,
				Source:         source.TypeError,
				Message:        fmt.Sprintf("error while importing %v: %v", item, depErr.Err),
				SuggestedFixes: fixes,
			})
			break
		}
	}
	return errors, nil
}

// missingPkgError returns an error message for a missing package that varies
// based on the user's workspace mode.
func missingPkgError(pkgPath string, moduleMode bool) error {
	// TODO(rfindley): improve this error. Previous versions of this error had
	// access to the full snapshot, and could provide more information (such as
	// the initialization error).
	if moduleMode {
		// Previously, we would present the initialization error here.
		return fmt.Errorf("no required module provides package %q", pkgPath)
	} else {
		// Previously, we would list the directories in GOROOT and GOPATH here.
		return fmt.Errorf("cannot find package %q in GOROOT or GOPATH", pkgPath)
	}
}

type extendedError struct {
	primary     types.Error
	secondaries []types.Error
}

func (e extendedError) Error() string {
	return e.primary.Error()
}

// expandErrors duplicates "secondary" errors by mapping them to their main
// error. Some errors returned by the type checker are followed by secondary
// errors which give more information about the error. These are errors in
// their own right, and they are marked by starting with \t. For instance, when
// there is a multiply-defined function, the secondary error points back to the
// definition first noticed.
//
// This function associates the secondary error with its primary error, which can
// then be used as RelatedInformation when the error becomes a diagnostic.
//
// If supportsRelatedInformation is false, the secondary is instead embedded as
// additional context in the primary error.
func expandErrors(errs []types.Error, supportsRelatedInformation bool) []extendedError {
	var result []extendedError
	for i := 0; i < len(errs); {
		original := extendedError{
			primary: errs[i],
		}
		for i++; i < len(errs); i++ {
			spl := errs[i]
			if len(spl.Msg) == 0 || spl.Msg[0] != '\t' {
				break
			}
			spl.Msg = spl.Msg[1:]
			original.secondaries = append(original.secondaries, spl)
		}

		// Clone the error to all its related locations -- VS Code, at least,
		// doesn't do it for us.
		result = append(result, original)
		for i, mainSecondary := range original.secondaries {
			// Create the new primary error, with a tweaked message, in the
			// secondary's location. We need to start from the secondary to
			// capture its unexported location fields.
			relocatedSecondary := mainSecondary
			if supportsRelatedInformation {
				relocatedSecondary.Msg = fmt.Sprintf("%v (see details)", original.primary.Msg)
			} else {
				relocatedSecondary.Msg = fmt.Sprintf("%v (this error: %v)", original.primary.Msg, mainSecondary.Msg)
			}
			relocatedSecondary.Soft = original.primary.Soft

			// Copy over the secondary errors, noting the location of the
			// current error we're cloning.
			clonedError := extendedError{primary: relocatedSecondary, secondaries: []types.Error{original.primary}}
			for j, secondary := range original.secondaries {
				if i == j {
					secondary.Msg += " (this error)"
				}
				clonedError.secondaries = append(clonedError.secondaries, secondary)
			}
			result = append(result, clonedError)
		}

	}
	return result
}

// An importFunc is an implementation of the single-method
// types.Importer interface based on a function value.
type importerFunc func(path string) (*types.Package, error)

func (f importerFunc) Import(path string) (*types.Package, error) { return f(path) }

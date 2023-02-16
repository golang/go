// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"go/ast"
	"go/types"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"

	"golang.org/x/mod/module"
	"golang.org/x/sync/errgroup"
	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/gopls/internal/lsp/source/methodsets"
	"golang.org/x/tools/gopls/internal/lsp/source/xrefs"
	"golang.org/x/tools/gopls/internal/span"
	"golang.org/x/tools/internal/bug"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/event/tag"
	"golang.org/x/tools/internal/memoize"
	"golang.org/x/tools/internal/packagesinternal"
	"golang.org/x/tools/internal/typeparams"
	"golang.org/x/tools/internal/typesinternal"
)

// A packageKey identifies a packageHandle in the snapshot.packages map.
type packageKey struct {
	mode source.ParseMode
	id   PackageID
}

type packageHandleKey source.Hash

// A packageHandle holds package information, some of which may not be fully
// evaluated.
//
// The only methods on packageHandle that are safe to call before calling await
// are Metadata and await itself.
type packageHandle struct {
	// TODO(rfindley): remove metadata from packageHandle. It is only used for
	// bug detection.
	m *source.Metadata

	// key is the hashed key for the package.
	//
	// It includes the all bits of the transitive closure of
	// dependencies's sources. This is more than type checking
	// really depends on: export data of direct deps should be
	// enough. (The key for analysis actions could similarly
	// hash only Facts of direct dependencies.)
	key packageHandleKey

	// The shared type-checking promise.
	promise *memoize.Promise // [typeCheckResult]
}

// typeCheckInputs contains the inputs of a call to typeCheckImpl, which
// type-checks a package.
type typeCheckInputs struct {
	id                       PackageID
	pkgPath                  PackagePath
	name                     PackageName
	mode                     source.ParseMode
	goFiles, compiledGoFiles []source.FileHandle
	sizes                    types.Sizes
	deps                     map[PackageID]*packageHandle
	depsByImpPath            map[ImportPath]PackageID
	goVersion                string // packages.Module.GoVersion, e.g. "1.18"
}

// typeCheckResult contains the result of a call to
// typeCheckImpl, which type-checks a package.
type typeCheckResult struct {
	pkg *syntaxPackage
	err error
}

// buildPackageHandle returns a handle for the future results of
// type-checking the package identified by id in the given mode.
// It assumes that the given ID already has metadata available, so it does not
// attempt to reload missing or invalid metadata. The caller must reload
// metadata if needed.
func (s *snapshot) buildPackageHandle(ctx context.Context, id PackageID, mode source.ParseMode) (*packageHandle, error) {
	packageKey := packageKey{id: id, mode: mode}

	s.mu.Lock()
	entry, hit := s.packages.Get(packageKey)
	m := s.meta.metadata[id]
	s.mu.Unlock()

	if m == nil {
		return nil, fmt.Errorf("no metadata for %s", id)
	}

	if hit {
		return entry.(*packageHandle), nil
	}

	// Begin computing the key by getting the depKeys for all dependencies.
	// This requires reading the transitive closure of dependencies' source files.
	//
	// It is tempting to parallelize the recursion here, but
	// without de-duplication of subtasks this would lead to an
	// exponential amount of work, and computing the key is
	// expensive as it reads all the source files transitively.
	// Notably, we don't update the s.packages cache until the
	// entire key has been computed.
	// TODO(adonovan): use a promise cache to ensure that the key
	// for each package is computed by at most one thread, then do
	// the recursive key building of dependencies in parallel.
	deps := make(map[PackageID]*packageHandle)
	for _, depID := range m.DepsByPkgPath {
		depHandle, err := s.buildPackageHandle(ctx, depID, s.workspaceParseMode(depID))
		// Don't use invalid metadata for dependencies if the top-level
		// metadata is valid. We only load top-level packages, so if the
		// top-level is valid, all of its dependencies should be as well.
		if err != nil {
			event.Error(ctx, fmt.Sprintf("%s: no dep handle for %s", id, depID), err, source.SnapshotLabels(s)...)

			// This check ensures we break out of the slow
			// buildPackageHandle recursion quickly when
			// context cancelation is detected within GetFile.
			if ctx.Err() != nil {
				return nil, ctx.Err() // cancelled
			}

			// One bad dependency should not prevent us from
			// checking the entire package. Leave depKeys[i] unset.
			continue
		}
		deps[depID] = depHandle
	}

	// Read both lists of files of this package, in parallel.
	//
	// goFiles aren't presented to the type checker--nor
	// are they included in the key, unsoundly--but their
	// syntax trees are available from (*pkg).File(URI).
	// TODO(adonovan): consider parsing them on demand?
	// The need should be rare.
	goFiles, compiledGoFiles, err := readGoFiles(ctx, s, m)
	if err != nil {
		return nil, err
	}

	goVersion := ""
	if m.Module != nil && m.Module.GoVersion != "" {
		goVersion = m.Module.GoVersion
	}

	inputs := typeCheckInputs{
		id:              m.ID,
		pkgPath:         m.PkgPath,
		name:            m.Name,
		mode:            mode,
		goFiles:         goFiles,
		compiledGoFiles: compiledGoFiles,
		sizes:           m.TypesSizes,
		deps:            deps,
		depsByImpPath:   m.DepsByImpPath,
		goVersion:       goVersion,
	}

	// All the file reading has now been done.
	// Create a handle for the result of type checking.
	phKey := computePackageKey(s, inputs)
	promise, release := s.store.Promise(phKey, func(ctx context.Context, arg interface{}) interface{} {
		pkg, err := typeCheckImpl(ctx, arg.(*snapshot), inputs)
		return typeCheckResult{pkg, err}
	})

	ph := &packageHandle{
		promise: promise,
		m:       m,
		key:     phKey,
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
	if prev, ok := s.packages.Get(packageKey); ok {
		prevPH := prev.(*packageHandle)
		release()
		if prevPH.m != ph.m {
			return nil, bug.Errorf("existing package handle does not match for %s", ph.m.ID)
		}
		return prevPH, nil
	}

	// Update the map.
	s.packages.Set(packageKey, ph, func(_, _ interface{}) { release() })

	return ph, nil
}

// readGoFiles reads the content of Metadata.GoFiles and
// Metadata.CompiledGoFiles, in parallel.
func readGoFiles(ctx context.Context, s *snapshot, m *source.Metadata) (goFiles, compiledGoFiles []source.FileHandle, err error) {
	var group errgroup.Group
	getFileHandles := func(files []span.URI) []source.FileHandle {
		fhs := make([]source.FileHandle, len(files))
		for i, uri := range files {
			i, uri := i, uri
			group.Go(func() (err error) {
				fhs[i], err = s.GetFile(ctx, uri) // ~25us
				return
			})
		}
		return fhs
	}
	return getFileHandles(m.GoFiles),
		getFileHandles(m.CompiledGoFiles),
		group.Wait()
}

func (s *snapshot) workspaceParseMode(id PackageID) source.ParseMode {
	s.mu.Lock()
	defer s.mu.Unlock()
	_, ws := s.workspacePackages[id]
	if !ws {
		return source.ParseExported
	}
	if s.view.Options().MemoryMode == source.ModeNormal {
		return source.ParseFull
	}
	if s.isActiveLocked(id) {
		return source.ParseFull
	}
	return source.ParseExported
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

	// parse mode
	fmt.Fprintf(hasher, "mode %d\n", inputs.mode)

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

	var hash [sha256.Size]byte
	hasher.Sum(hash[:0])
	return packageHandleKey(hash)
}

// await waits for typeCheckImpl to complete and returns its result.
func (ph *packageHandle) await(ctx context.Context, s *snapshot) (*syntaxPackage, error) {
	v, err := s.awaitPromise(ctx, ph.promise)
	if err != nil {
		return nil, err
	}
	data := v.(typeCheckResult)
	return data.pkg, data.err
}

func (ph *packageHandle) cached() (*syntaxPackage, error) {
	v := ph.promise.Cached()
	if v == nil {
		return nil, fmt.Errorf("no cached type information for %s", ph.m.PkgPath)
	}
	data := v.(typeCheckResult)
	return data.pkg, data.err
}

// typeCheckImpl type checks the parsed source files in compiledGoFiles.
// (The resulting pkg also holds the parsed but not type-checked goFiles.)
// deps holds the future results of type-checking the direct dependencies.
func typeCheckImpl(ctx context.Context, snapshot *snapshot, inputs typeCheckInputs) (*syntaxPackage, error) {
	// Start type checking of direct dependencies,
	// in parallel and asynchronously.
	// As the type checker imports each of these
	// packages, it will wait for its completion.
	var wg sync.WaitGroup
	for _, dep := range inputs.deps {
		wg.Add(1)
		go func(dep *packageHandle) {
			dep.await(ctx, snapshot) // ignore result
			wg.Done()
		}(dep)
	}
	// The 'defer' below is unusual but intentional:
	// it is not necessary that each call to dep.check
	// complete before type checking begins, as the type
	// checker will wait for those it needs. But they do
	// need to complete before this function returns and
	// the snapshot is possibly destroyed.
	defer wg.Wait()

	var filter *unexportedFilter
	if inputs.mode == source.ParseExported {
		filter = &unexportedFilter{uses: map[string]bool{}}
	}
	pkg, err := doTypeCheck(ctx, snapshot, inputs, filter)
	if err != nil {
		return nil, err
	}

	if inputs.mode == source.ParseExported {
		// The AST filtering is a little buggy and may remove things it
		// shouldn't. If we only got undeclared name errors, try one more
		// time keeping those names.
		missing, unexpected := filter.ProcessErrors(pkg.typeErrors)
		if len(unexpected) == 0 && len(missing) != 0 {
			pkg, err = doTypeCheck(ctx, snapshot, inputs, filter)
			if err != nil {
				return nil, err
			}
			missing, unexpected = filter.ProcessErrors(pkg.typeErrors)
		}
		if len(unexpected) != 0 || len(missing) != 0 {
			pkg, err = doTypeCheck(ctx, snapshot, inputs, nil)
			if err != nil {
				return nil, err
			}
		}
	}

	// We don't care about a package's errors unless we have parsed it in full.
	if inputs.mode != source.ParseFull {
		return pkg, nil
	}

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
		diags, err := parseErrorDiagnostics(snapshot, pkg, e)
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
	for _, e := range expandErrors(unexpanded, snapshot.View().Options().RelatedInformationSupported) {
		diags, err := typeErrorDiagnostics(snapshot, pkg, e)
		if err != nil {
			event.Error(ctx, "unable to compute positions for type errors", err, tag.Package.Of(string(inputs.id)))
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

func doTypeCheck(ctx context.Context, snapshot *snapshot, inputs typeCheckInputs, astFilter *unexportedFilter) (*syntaxPackage, error) {
	ctx, done := event.Start(ctx, "cache.typeCheck", tag.Package.Of(string(inputs.id)))
	defer done()

	pkg := &syntaxPackage{
		id:        inputs.id,
		mode:      inputs.mode,
		fset:      snapshot.view.fset, // must match parse call below (snapshot.ParseGo for now)
		types:     types.NewPackage(string(inputs.pkgPath), string(inputs.name)),
		importMap: new(importMap),
		typesInfo: &types.Info{
			Types:      make(map[ast.Expr]types.TypeAndValue),
			Defs:       make(map[*ast.Ident]types.Object),
			Uses:       make(map[*ast.Ident]types.Object),
			Implicits:  make(map[ast.Node]types.Object),
			Selections: make(map[*ast.SelectorExpr]*types.Selection),
			Scopes:     make(map[ast.Node]*types.Scope),
		},
	}
	typeparams.InitInstanceInfo(pkg.typesInfo)
	defer func() { pkg.importMap.types = pkg.types }() // simplifies early return in "unsafe"

	// Parse the non-compiled GoFiles. (These aren't presented to
	// the type checker but are part of the returned pkg.)
	// TODO(adonovan): opt: parallelize parsing.
	for _, fh := range inputs.goFiles {
		goMode := inputs.mode
		if inputs.mode == source.ParseExported {
			// This package is being loaded only for type information,
			// to which non-compiled Go files are irrelevant,
			// so parse only the header.
			goMode = source.ParseHeader
		}
		pgf, err := snapshot.ParseGo(ctx, fh, goMode)
		if err != nil {
			return nil, err
		}
		pkg.goFiles = append(pkg.goFiles, pgf)
	}

	// Parse the CompiledGoFiles: those seen by the compiler/typechecker.
	if err := parseCompiledGoFiles(ctx, inputs.compiledGoFiles, snapshot, inputs.mode, pkg, astFilter); err != nil {
		return nil, err
	}

	// Use the default type information for the unsafe package.
	if inputs.pkgPath == "unsafe" {
		// Don't type check Unsafe: it's unnecessary, and doing so exposes a data
		// race to Unsafe.completed.
		// TODO(adonovan): factor (tail-merge) with the normal control path.
		pkg.types = types.Unsafe
		pkg.methodsets = methodsets.NewIndex(pkg.fset, pkg.types)
		pkg.xrefs = xrefs.Index(pkg.compiledGoFiles, pkg.types, pkg.typesInfo)
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

	cfg := &types.Config{
		Sizes: inputs.sizes,
		Error: func(e error) {
			pkg.typeErrors = append(pkg.typeErrors, e.(types.Error))
		},
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
			dep, ok := inputs.deps[id] // id may be ""
			if !ok {
				return nil, snapshot.missingPkgError(path)
			}
			if !source.IsValidImport(inputs.pkgPath, dep.m.PkgPath) {
				return nil, fmt.Errorf("invalid use of internal package %s", path)
			}
			depPkg, err := dep.await(ctx, snapshot)
			if err != nil {
				return nil, err
			}
			pkg.importMap.union(depPkg.importMap)
			return depPkg.types, nil
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

	if inputs.mode != source.ParseFull {
		cfg.DisableUnusedImportCheck = true
		cfg.IgnoreFuncBodies = true
	}

	// We want to type check cgo code if go/types supports it.
	// We passed typecheckCgo to go/packages when we Loaded.
	typesinternal.SetUsesCgo(cfg)

	check := types.NewChecker(cfg, pkg.fset, pkg.types, pkg.typesInfo)

	var files []*ast.File
	for _, cgf := range pkg.compiledGoFiles {
		files = append(files, cgf.File)
	}

	// Type checking errors are handled via the config, so ignore them here.
	_ = check.Files(files) // 50us-15ms, depending on size of package

	// Build global index of method sets for 'implementations' queries.
	pkg.methodsets = methodsets.NewIndex(pkg.fset, pkg.types)

	// Build global index of outbound cross-references.
	pkg.xrefs = xrefs.Index(pkg.compiledGoFiles, pkg.types, pkg.typesInfo)

	// If the context was cancelled, we may have returned a ton of transient
	// errors to the type checker. Swallow them.
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}
	return pkg, nil
}

func parseCompiledGoFiles(ctx context.Context, compiledGoFiles []source.FileHandle, snapshot *snapshot, mode source.ParseMode, pkg *syntaxPackage, astFilter *unexportedFilter) error {
	// TODO(adonovan): opt: parallelize this loop, which takes 1-25ms.
	for _, fh := range compiledGoFiles {
		var pgf *source.ParsedGoFile
		var err error
		// Only parse Full through the cache -- we need to own Exported ASTs
		// to prune them.
		if mode == source.ParseFull {
			pgf, err = snapshot.ParseGo(ctx, fh, mode)
		} else {
			pgf, err = parseGoImpl(ctx, pkg.fset, fh, mode) // ~20us/KB
		}
		if err != nil {
			return err
		}
		pkg.compiledGoFiles = append(pkg.compiledGoFiles, pgf)
		if pgf.ParseErr != nil {
			pkg.parseErrors = append(pkg.parseErrors, pgf.ParseErr)
		}
		// If we have fixed parse errors in any of the files, we should hide type
		// errors, as they may be completely nonsensical.
		pkg.hasFixedFiles = pkg.hasFixedFiles || pgf.Fixed
	}

	// Optionally remove parts that don't affect the exported API.
	if mode == source.ParseExported {
		// TODO(adonovan): opt: experiment with pre-parser
		// trimming, either a scanner-based implementation
		// such as https://go.dev/play/p/KUrObH1YkX8 (~31%
		// speedup), or a byte-oriented implementation (2x
		// speedup).
		if astFilter != nil {
			// aggressive pruning based on reachability
			var files []*ast.File
			for _, cgf := range pkg.compiledGoFiles {
				files = append(files, cgf.File)
			}
			astFilter.Filter(files)
		} else {
			// simple trimming of function bodies
			for _, cgf := range pkg.compiledGoFiles {
				trimAST(cgf.File)
			}
		}
	}

	return nil
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
		fset := source.SingletonFileSet(pgf.Tok)
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
func (s *snapshot) missingPkgError(pkgPath string) error {
	var b strings.Builder
	if s.workspaceMode()&moduleMode == 0 {
		gorootSrcPkg := filepath.FromSlash(filepath.Join(s.view.goroot, "src", pkgPath))
		fmt.Fprintf(&b, "cannot find package %q in any of \n\t%s (from $GOROOT)", pkgPath, gorootSrcPkg)
		for _, gopath := range filepath.SplitList(s.view.gopath) {
			gopathSrcPkg := filepath.FromSlash(filepath.Join(gopath, "src", pkgPath))
			fmt.Fprintf(&b, "\n\t%s (from $GOPATH)", gopathSrcPkg)
		}
	} else {
		fmt.Fprintf(&b, "no required module provides package %q", pkgPath)
		if err := s.getInitializationError(); err != nil {
			fmt.Fprintf(&b, "\n(workspace configuration error: %s)", err.MainError)
		}
	}
	return errors.New(b.String())
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

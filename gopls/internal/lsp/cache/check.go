// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"go/ast"
	"go/types"
	"path/filepath"
	"regexp"
	"strings"
	"sync"

	"golang.org/x/mod/module"
	"golang.org/x/sync/errgroup"
	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/source"
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

// A packageHandle is a handle to the future result of type-checking a package.
// The resulting package is obtained from the await() method.
type packageHandle struct {
	promise *memoize.Promise // [typeCheckResult]

	// m is the metadata associated with the package.
	m *KnownMetadata

	// key is the hashed key for the package.
	//
	// It includes the all bits of the transitive closure of
	// dependencies's sources. This is more than type checking
	// really depends on: export data of direct deps should be
	// enough. (The key for analysis actions could similarly
	// hash only Facts of direct dependencies.)
	key packageHandleKey
}

// typeCheckResult contains the result of a call to
// typeCheckImpl, which type-checks a package.
type typeCheckResult struct {
	pkg *pkg
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
	var depKey source.Hash // XOR of all unique deps
	for _, depID := range m.Imports {
		depHandle, ok := deps[depID]
		if ok {
			continue // e.g. duplicate import
		}
		depHandle, err := s.buildPackageHandle(ctx, depID, s.workspaceParseMode(depID))
		// Don't use invalid metadata for dependencies if the top-level
		// metadata is valid. We only load top-level packages, so if the
		// top-level is valid, all of its dependencies should be as well.
		if err != nil || m.Valid && !depHandle.m.Valid {
			if err != nil {
				event.Error(ctx, fmt.Sprintf("%s: no dep handle for %s", id, depID), err, tag.Snapshot.Of(s.id))
			} else {
				event.Log(ctx, fmt.Sprintf("%s: invalid dep handle for %s", id, depID), tag.Snapshot.Of(s.id))
			}

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

		depKey.XORWith(source.Hash(depHandle.key))

		deps[depID] = depHandle
	}

	// Read both lists of files of this package, in parallel.
	//
	// goFiles aren't presented to the type checker--nor
	// are they included in the key, unsoundly--but their
	// syntax trees are available from (*pkg).File(URI).
	// TODO(adonovan): consider parsing them on demand?
	// The need should be rare.
	goFiles, compiledGoFiles, err := readGoFiles(ctx, s, m.Metadata)
	if err != nil {
		return nil, err
	}

	// All the file reading has now been done.
	// Create a handle for the result of type checking.
	experimentalKey := s.View().Options().ExperimentalPackageCacheKey
	phKey := computePackageKey(m.ID, compiledGoFiles, m, depKey, mode, experimentalKey)
	promise, release := s.store.Promise(phKey, func(ctx context.Context, arg interface{}) interface{} {
		pkg, err := typeCheckImpl(ctx, arg.(*snapshot), goFiles, compiledGoFiles, m.Metadata, mode, deps)
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
	if s.meta.metadata[ph.m.ID].Metadata != ph.m.Metadata {
		return nil, fmt.Errorf("stale metadata for %s", ph.m.ID)
	}

	// Check cache again in case another thread got there first.
	if prev, ok := s.packages.Get(packageKey); ok {
		prevPH := prev.(*packageHandle)
		release()
		if prevPH.m.Metadata != ph.m.Metadata {
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
func readGoFiles(ctx context.Context, s *snapshot, m *Metadata) (goFiles, compiledGoFiles []source.FileHandle, err error) {
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
func computePackageKey(id PackageID, files []source.FileHandle, m *KnownMetadata, depsKey source.Hash, mode source.ParseMode, experimentalKey bool) packageHandleKey {
	// TODO(adonovan): opt: no need to materalize the bytes; hash them directly.
	// Also, use field separators to avoid spurious collisions.
	b := bytes.NewBuffer(nil)
	b.WriteString(string(id))
	if m.Module != nil {
		b.WriteString(m.Module.GoVersion) // go version affects type check errors.
	}
	if !experimentalKey {
		// cfg was used to produce the other hashed inputs (package ID, parsed Go
		// files, and deps). It should not otherwise affect the inputs to the type
		// checker, so this experiment omits it. This should increase cache hits on
		// the daemon as cfg contains the environment and working directory.
		hc := hashConfig(m.Config)
		b.Write(hc[:])
	}
	b.WriteByte(byte(mode))
	b.Write(depsKey[:])
	for _, file := range files {
		b.WriteString(file.FileIdentity().String())
	}
	// Metadata errors are interpreted and memoized on the computed package, so
	// we must hash them into the key here.
	//
	// TODO(rfindley): handle metadata diagnostics independently from
	// type-checking diagnostics.
	for _, err := range m.Errors {
		b.WriteString(err.Msg)
		b.WriteString(err.Pos)
		b.WriteRune(rune(err.Kind))
	}
	return packageHandleKey(source.HashOf(b.Bytes()))
}

// hashConfig returns the hash for the *packages.Config.
func hashConfig(config *packages.Config) source.Hash {
	// TODO(adonovan): opt: don't materialize the bytes; hash them directly.
	// Also, use sound field separators to avoid collisions.
	var b bytes.Buffer

	// Dir, Mode, Env, BuildFlags are the parts of the config that can change.
	b.WriteString(config.Dir)
	b.WriteRune(rune(config.Mode))

	for _, e := range config.Env {
		b.WriteString(e)
	}
	for _, f := range config.BuildFlags {
		b.WriteString(f)
	}
	return source.HashOf(b.Bytes())
}

// await waits for typeCheckImpl to complete and returns its result.
func (ph *packageHandle) await(ctx context.Context, s *snapshot) (*pkg, error) {
	v, err := s.awaitPromise(ctx, ph.promise)
	if err != nil {
		return nil, err
	}
	data := v.(typeCheckResult)
	return data.pkg, data.err
}

func (ph *packageHandle) CompiledGoFiles() []span.URI {
	return ph.m.CompiledGoFiles
}

func (ph *packageHandle) ID() string {
	return string(ph.m.ID)
}

func (ph *packageHandle) cached() (*pkg, error) {
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
func typeCheckImpl(ctx context.Context, snapshot *snapshot, goFiles, compiledGoFiles []source.FileHandle, m *Metadata, mode source.ParseMode, deps map[PackageID]*packageHandle) (*pkg, error) {
	// Start type checking of direct dependencies,
	// in parallel and asynchronously.
	// As the type checker imports each of these
	// packages, it will wait for its completion.
	var wg sync.WaitGroup
	for _, dep := range deps {
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
	if mode == source.ParseExported {
		filter = &unexportedFilter{uses: map[string]bool{}}
	}
	pkg, err := doTypeCheck(ctx, snapshot, goFiles, compiledGoFiles, m, mode, deps, filter)
	if err != nil {
		return nil, err
	}

	if mode == source.ParseExported {
		// The AST filtering is a little buggy and may remove things it
		// shouldn't. If we only got undeclared name errors, try one more
		// time keeping those names.
		missing, unexpected := filter.ProcessErrors(pkg.typeErrors)
		if len(unexpected) == 0 && len(missing) != 0 {
			pkg, err = doTypeCheck(ctx, snapshot, goFiles, compiledGoFiles, m, mode, deps, filter)
			if err != nil {
				return nil, err
			}
			missing, unexpected = filter.ProcessErrors(pkg.typeErrors)
		}
		if len(unexpected) != 0 || len(missing) != 0 {
			pkg, err = doTypeCheck(ctx, snapshot, goFiles, compiledGoFiles, m, mode, deps, nil)
			if err != nil {
				return nil, err
			}
		}
	}
	// If this is a replaced module in the workspace, the version is
	// meaningless, and we don't want clients to access it.
	if m.Module != nil {
		version := m.Module.Version
		if source.IsWorkspaceModuleVersion(version) {
			version = ""
		}
		pkg.version = &module.Version{
			Path:    m.Module.Path,
			Version: version,
		}
	}

	// We don't care about a package's errors unless we have parsed it in full.
	if mode != source.ParseFull {
		return pkg, nil
	}

	for _, e := range m.Errors {
		diags, err := goPackagesErrorDiagnostics(snapshot, pkg, e)
		if err != nil {
			event.Error(ctx, "unable to compute positions for list errors", err, tag.Package.Of(pkg.ID()))
			continue
		}
		pkg.diagnostics = append(pkg.diagnostics, diags...)
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
			event.Error(ctx, "unable to compute positions for parse errors", err, tag.Package.Of(pkg.ID()))
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
			event.Error(ctx, "unable to compute positions for type errors", err, tag.Package.Of(pkg.ID()))
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

	depsErrors, err := snapshot.depsErrors(ctx, pkg)
	if err != nil {
		return nil, err
	}
	pkg.diagnostics = append(pkg.diagnostics, depsErrors...)

	return pkg, nil
}

var goVersionRx = regexp.MustCompile(`^go([1-9][0-9]*)\.(0|[1-9][0-9]*)$`)

func doTypeCheck(ctx context.Context, snapshot *snapshot, goFiles, compiledGoFiles []source.FileHandle, m *Metadata, mode source.ParseMode, deps map[PackageID]*packageHandle, astFilter *unexportedFilter) (*pkg, error) {
	ctx, done := event.Start(ctx, "cache.typeCheck", tag.Package.Of(string(m.ID)))
	defer done()

	pkg := &pkg{
		m:             m,
		mode:          mode,
		depsByPkgPath: make(map[PackagePath]*pkg),
		types:         types.NewPackage(string(m.PkgPath), string(m.Name)),
		typesInfo: &types.Info{
			Types:      make(map[ast.Expr]types.TypeAndValue),
			Defs:       make(map[*ast.Ident]types.Object),
			Uses:       make(map[*ast.Ident]types.Object),
			Implicits:  make(map[ast.Node]types.Object),
			Selections: make(map[*ast.SelectorExpr]*types.Selection),
			Scopes:     make(map[ast.Node]*types.Scope),
		},
		typesSizes: m.TypesSizes,
	}
	typeparams.InitInstanceInfo(pkg.typesInfo)

	// In the presence of line directives, we may need to report errors in
	// non-compiled Go files, so we need to register them on the package.
	// However, we only need to really parse them in ParseFull mode, when
	// the user might actually be looking at the file.
	goMode := source.ParseFull
	if mode != source.ParseFull {
		goMode = source.ParseHeader
	}

	// Parse the GoFiles. (These aren't presented to the type
	// checker but are part of the returned pkg.)
	// TODO(adonovan): opt: parallelize parsing.
	for _, fh := range goFiles {
		pgf, err := snapshot.ParseGo(ctx, fh, goMode)
		if err != nil {
			return nil, err
		}
		pkg.goFiles = append(pkg.goFiles, pgf)
	}

	// Parse the CompiledGoFiles: those seen by the compiler/typechecker.
	if err := parseCompiledGoFiles(ctx, compiledGoFiles, snapshot, mode, pkg, astFilter); err != nil {
		return nil, err
	}

	// Use the default type information for the unsafe package.
	if m.PkgPath == "unsafe" {
		// Don't type check Unsafe: it's unnecessary, and doing so exposes a data
		// race to Unsafe.completed.
		pkg.types = types.Unsafe
		return pkg, nil
	}

	if len(m.CompiledGoFiles) == 0 {
		// No files most likely means go/packages failed. Try to attach error
		// messages to the file as much as possible.
		var found bool
		for _, e := range m.Errors {
			srcDiags, err := goPackagesErrorDiagnostics(snapshot, pkg, e)
			if err != nil {
				continue
			}
			found = true
			pkg.diagnostics = append(pkg.diagnostics, srcDiags...)
		}
		if found {
			return pkg, nil
		}
		return nil, fmt.Errorf("no parsed files for package %s, expected: %v, errors: %v", pkg.m.PkgPath, pkg.compiledGoFiles, m.Errors)
	}

	cfg := &types.Config{
		Error: func(e error) {
			pkg.typeErrors = append(pkg.typeErrors, e.(types.Error))
		},
		Importer: importerFunc(func(path string) (*types.Package, error) {
			// While all of the import errors could be reported
			// based on the metadata before we start type checking,
			// reporting them via types.Importer places the errors
			// at the correct source location.
			id, ok := pkg.m.Imports[ImportPath(path)]
			if !ok {
				// If the import declaration is broken,
				// go list may fail to report metadata about it.
				// See TestFixImportDecl for an example.
				return nil, fmt.Errorf("missing metadata for import of %q", path)
			}
			dep, ok := deps[id]
			if !ok {
				return nil, snapshot.missingPkgError(ctx, path)
			}
			if !source.IsValidImport(string(m.PkgPath), string(dep.m.PkgPath)) {
				return nil, fmt.Errorf("invalid use of internal package %s", path)
			}
			depPkg, err := dep.await(ctx, snapshot)
			if err != nil {
				return nil, err
			}
			pkg.depsByPkgPath[depPkg.m.PkgPath] = depPkg
			return depPkg.types, nil
		}),
	}
	if pkg.m.Module != nil && pkg.m.Module.GoVersion != "" {
		goVersion := "go" + pkg.m.Module.GoVersion
		// types.NewChecker panics if GoVersion is invalid. An unparsable mod
		// file should probably stop us before we get here, but double check
		// just in case.
		if goVersionRx.MatchString(goVersion) {
			typesinternal.SetGoVersion(cfg, goVersion)
		}
	}

	if mode != source.ParseFull {
		cfg.DisableUnusedImportCheck = true
		cfg.IgnoreFuncBodies = true
	}

	// We want to type check cgo code if go/types supports it.
	// We passed typecheckCgo to go/packages when we Loaded.
	typesinternal.SetUsesCgo(cfg)

	check := types.NewChecker(cfg, snapshot.FileSet(), pkg.types, pkg.typesInfo)

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

func parseCompiledGoFiles(ctx context.Context, compiledGoFiles []source.FileHandle, snapshot *snapshot, mode source.ParseMode, pkg *pkg, astFilter *unexportedFilter) error {
	// TODO(adonovan): opt: parallelize this loop, which takes 1-25ms.
	for _, fh := range compiledGoFiles {
		var pgf *source.ParsedGoFile
		var err error
		// Only parse Full through the cache -- we need to own Exported ASTs
		// to prune them.
		if mode == source.ParseFull {
			pgf, err = snapshot.ParseGo(ctx, fh, mode)
		} else {
			pgf, err = parseGoImpl(ctx, snapshot.FileSet(), fh, mode) // ~20us/KB
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

func (s *snapshot) depsErrors(ctx context.Context, pkg *pkg) ([]*source.Diagnostic, error) {
	// Select packages that can't be found, and were imported in non-workspace packages.
	// Workspace packages already show their own errors.
	var relevantErrors []*packagesinternal.PackageError
	for _, depsError := range pkg.m.depsErrors {
		// Up to Go 1.15, the missing package was included in the stack, which
		// was presumably a bug. We want the next one up.
		directImporterIdx := len(depsError.ImportStack) - 1
		if s.view.goversion < 15 {
			directImporterIdx = len(depsError.ImportStack) - 2
		}
		if directImporterIdx < 0 {
			continue
		}

		directImporter := depsError.ImportStack[directImporterIdx]
		if s.isWorkspacePackage(PackageID(directImporter)) {
			continue
		}
		relevantErrors = append(relevantErrors, depsError)
	}

	// Don't build the import index for nothing.
	if len(relevantErrors) == 0 {
		return nil, nil
	}

	// Build an index of all imports in the package.
	type fileImport struct {
		cgf *source.ParsedGoFile
		imp *ast.ImportSpec
	}
	allImports := map[string][]fileImport{}
	for _, cgf := range pkg.compiledGoFiles {
		for _, group := range astutil.Imports(s.FileSet(), cgf.File) {
			for _, imp := range group {
				if imp.Path == nil {
					continue
				}
				path := strings.Trim(imp.Path.Value, `"`)
				allImports[path] = append(allImports[path], fileImport{cgf, imp})
			}
		}
	}

	// Apply a diagnostic to any import involved in the error, stopping once
	// we reach the workspace.
	var errors []*source.Diagnostic
	for _, depErr := range relevantErrors {
		for i := len(depErr.ImportStack) - 1; i >= 0; i-- {
			item := depErr.ImportStack[i]
			if s.isWorkspacePackage(PackageID(item)) {
				break
			}

			for _, imp := range allImports[item] {
				rng, err := source.NewMappedRange(imp.cgf.Tok, imp.cgf.Mapper, imp.imp.Pos(), imp.imp.End()).Range()
				if err != nil {
					return nil, err
				}
				fixes, err := goGetQuickFixes(s, imp.cgf.URI, item)
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

	if len(pkg.compiledGoFiles) == 0 {
		return errors, nil
	}
	mod := s.GoModForFile(pkg.compiledGoFiles[0].URI)
	if mod == "" {
		return errors, nil
	}
	fh, err := s.GetFile(ctx, mod)
	if err != nil {
		return nil, err
	}
	pm, err := s.ParseMod(ctx, fh)
	if err != nil {
		return nil, err
	}

	// Add a diagnostic to the module that contained the lowest-level import of
	// the missing package.
	for _, depErr := range relevantErrors {
		for i := len(depErr.ImportStack) - 1; i >= 0; i-- {
			item := depErr.ImportStack[i]
			m := s.getMetadata(PackageID(item))
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
			fixes, err := goGetQuickFixes(s, pm.URI, item)
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
func (s *snapshot) missingPkgError(ctx context.Context, pkgPath string) error {
	var b strings.Builder
	if s.workspaceMode()&moduleMode == 0 {
		gorootSrcPkg := filepath.FromSlash(filepath.Join(s.view.goroot, "src", pkgPath))

		b.WriteString(fmt.Sprintf("cannot find package %q in any of \n\t%s (from $GOROOT)", pkgPath, gorootSrcPkg))

		for _, gopath := range filepath.SplitList(s.view.gopath) {
			gopathSrcPkg := filepath.FromSlash(filepath.Join(gopath, "src", pkgPath))
			b.WriteString(fmt.Sprintf("\n\t%s (from $GOPATH)", gopathSrcPkg))
		}
	} else {
		b.WriteString(fmt.Sprintf("no required module provides package %q", pkgPath))
		if err := s.getInitializationError(ctx); err != nil {
			b.WriteString(fmt.Sprintf("(workspace configuration error: %s)", err.MainError))
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

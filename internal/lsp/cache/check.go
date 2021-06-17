// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"bytes"
	"context"
	"fmt"
	"go/ast"
	"go/types"
	"path"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	"golang.org/x/mod/module"
	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/lsp/debug/tag"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/memoize"
	"golang.org/x/tools/internal/packagesinternal"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/typeparams"
	"golang.org/x/tools/internal/typesinternal"
	errors "golang.org/x/xerrors"
)

type packageHandleKey string

type packageHandle struct {
	handle *memoize.Handle

	goFiles, compiledGoFiles []*parseGoHandle

	// mode is the mode the files were parsed in.
	mode source.ParseMode

	// m is the metadata associated with the package.
	m *knownMetadata

	// key is the hashed key for the package.
	key packageHandleKey
}

func (ph *packageHandle) packageKey() packageKey {
	return packageKey{
		id:   ph.m.id,
		mode: ph.mode,
	}
}

func (ph *packageHandle) imports(ctx context.Context, s source.Snapshot) (result []string) {
	for _, pgh := range ph.goFiles {
		f, err := s.ParseGo(ctx, pgh.file, source.ParseHeader)
		if err != nil {
			continue
		}
		seen := map[string]struct{}{}
		for _, impSpec := range f.File.Imports {
			imp := strings.Trim(impSpec.Path.Value, `"`)
			if _, ok := seen[imp]; !ok {
				seen[imp] = struct{}{}
				result = append(result, imp)
			}
		}
	}

	sort.Strings(result)
	return result
}

// packageData contains the data produced by type-checking a package.
type packageData struct {
	pkg *pkg
	err error
}

// buildPackageHandle returns a packageHandle for a given package and mode.
// It assumes that the given ID already has metadata available, so it does not
// attempt to reload missing or invalid metadata. The caller must reload
// metadata if needed.
func (s *snapshot) buildPackageHandle(ctx context.Context, id packageID, mode source.ParseMode) (*packageHandle, error) {
	if ph := s.getPackage(id, mode); ph != nil {
		return ph, nil
	}

	// Build the packageHandle for this ID and its dependencies.
	ph, deps, err := s.buildKey(ctx, id, mode)
	if err != nil {
		return nil, err
	}

	// Do not close over the packageHandle or the snapshot in the Bind function.
	// This creates a cycle, which causes the finalizers to never run on the handles.
	// The possible cycles are:
	//
	//     packageHandle.h.function -> packageHandle
	//     packageHandle.h.function -> snapshot -> packageHandle
	//

	m := ph.m
	key := ph.key

	h := s.generation.Bind(key, func(ctx context.Context, arg memoize.Arg) interface{} {
		snapshot := arg.(*snapshot)

		// Begin loading the direct dependencies, in parallel.
		var wg sync.WaitGroup
		for _, dep := range deps {
			wg.Add(1)
			go func(dep *packageHandle) {
				dep.check(ctx, snapshot)
				wg.Done()
			}(dep)
		}

		data := &packageData{}
		data.pkg, data.err = typeCheck(ctx, snapshot, m.metadata, mode, deps)
		// Make sure that the workers above have finished before we return,
		// especially in case of cancellation.
		wg.Wait()

		return data
	}, nil)
	ph.handle = h

	// Cache the handle in the snapshot. If a package handle has already
	// been cached, addPackage will return the cached value. This is fine,
	// since the original package handle above will have no references and be
	// garbage collected.
	ph = s.addPackageHandle(ph)

	return ph, nil
}

// buildKey computes the key for a given packageHandle.
func (s *snapshot) buildKey(ctx context.Context, id packageID, mode source.ParseMode) (*packageHandle, map[packagePath]*packageHandle, error) {
	m := s.getMetadata(id)
	if m == nil {
		return nil, nil, errors.Errorf("no metadata for %s", id)
	}
	goFiles, err := s.parseGoHandles(ctx, m.goFiles, mode)
	if err != nil {
		return nil, nil, err
	}
	compiledGoFiles, err := s.parseGoHandles(ctx, m.compiledGoFiles, mode)
	if err != nil {
		return nil, nil, err
	}
	ph := &packageHandle{
		m:               m,
		goFiles:         goFiles,
		compiledGoFiles: compiledGoFiles,
		mode:            mode,
	}
	// Make sure all of the depList are sorted.
	depList := append([]packageID{}, m.deps...)
	sort.Slice(depList, func(i, j int) bool {
		return depList[i] < depList[j]
	})

	deps := make(map[packagePath]*packageHandle)

	// Begin computing the key by getting the depKeys for all dependencies.
	var depKeys []packageHandleKey
	for _, depID := range depList {
		depHandle, err := s.buildPackageHandle(ctx, depID, s.workspaceParseMode(depID))
		// Don't use invalid metadata for dependencies if the top-level
		// metadata is valid. We only load top-level packages, so if the
		// top-level is valid, all of its dependencies should be as well.
		if err != nil || m.valid && !depHandle.m.valid {
			if err != nil {
				event.Error(ctx, fmt.Sprintf("%s: no dep handle for %s", id, depID), err, tag.Snapshot.Of(s.id))
			} else {
				event.Log(ctx, fmt.Sprintf("%s: invalid dep handle for %s", id, depID), tag.Snapshot.Of(s.id))
			}

			if ctx.Err() != nil {
				return nil, nil, ctx.Err()
			}
			// One bad dependency should not prevent us from checking the entire package.
			// Add a special key to mark a bad dependency.
			depKeys = append(depKeys, packageHandleKey(fmt.Sprintf("%s import not found", depID)))
			continue
		}
		deps[depHandle.m.pkgPath] = depHandle
		depKeys = append(depKeys, depHandle.key)
	}
	experimentalKey := s.View().Options().ExperimentalPackageCacheKey
	ph.key = checkPackageKey(ph.m.id, compiledGoFiles, m.config, depKeys, mode, experimentalKey)
	return ph, deps, nil
}

func (s *snapshot) workspaceParseMode(id packageID) source.ParseMode {
	s.mu.Lock()
	defer s.mu.Unlock()
	_, ws := s.workspacePackages[id]
	if !ws {
		return source.ParseExported
	}
	if s.view.Options().MemoryMode == source.ModeNormal {
		return source.ParseFull
	}

	// Degraded mode. Check for open files.
	m, ok := s.metadata[id]
	if !ok {
		return source.ParseExported
	}
	for _, cgf := range m.compiledGoFiles {
		if s.isOpenLocked(cgf) {
			return source.ParseFull
		}
	}
	return source.ParseExported
}

func checkPackageKey(id packageID, pghs []*parseGoHandle, cfg *packages.Config, deps []packageHandleKey, mode source.ParseMode, experimentalKey bool) packageHandleKey {
	b := bytes.NewBuffer(nil)
	b.WriteString(string(id))
	if !experimentalKey {
		// cfg was used to produce the other hashed inputs (package ID, parsed Go
		// files, and deps). It should not otherwise affect the inputs to the type
		// checker, so this experiment omits it. This should increase cache hits on
		// the daemon as cfg contains the environment and working directory.
		b.WriteString(hashConfig(cfg))
	}
	b.WriteByte(byte(mode))
	for _, dep := range deps {
		b.WriteString(string(dep))
	}
	for _, cgf := range pghs {
		b.WriteString(cgf.file.FileIdentity().String())
	}
	return packageHandleKey(hashContents(b.Bytes()))
}

// hashEnv returns a hash of the snapshot's configuration.
func hashEnv(s *snapshot) string {
	s.view.optionsMu.Lock()
	env := s.view.options.EnvSlice()
	s.view.optionsMu.Unlock()

	b := &bytes.Buffer{}
	for _, e := range env {
		b.WriteString(e)
	}
	return hashContents(b.Bytes())
}

// hashConfig returns the hash for the *packages.Config.
func hashConfig(config *packages.Config) string {
	b := bytes.NewBuffer(nil)

	// Dir, Mode, Env, BuildFlags are the parts of the config that can change.
	b.WriteString(config.Dir)
	b.WriteString(string(rune(config.Mode)))

	for _, e := range config.Env {
		b.WriteString(e)
	}
	for _, f := range config.BuildFlags {
		b.WriteString(f)
	}
	return hashContents(b.Bytes())
}

func (ph *packageHandle) Check(ctx context.Context, s source.Snapshot) (source.Package, error) {
	return ph.check(ctx, s.(*snapshot))
}

func (ph *packageHandle) check(ctx context.Context, s *snapshot) (*pkg, error) {
	v, err := ph.handle.Get(ctx, s.generation, s)
	if err != nil {
		return nil, err
	}
	data := v.(*packageData)
	return data.pkg, data.err
}

func (ph *packageHandle) CompiledGoFiles() []span.URI {
	return ph.m.compiledGoFiles
}

func (ph *packageHandle) ID() string {
	return string(ph.m.id)
}

func (ph *packageHandle) cached(g *memoize.Generation) (*pkg, error) {
	v := ph.handle.Cached(g)
	if v == nil {
		return nil, errors.Errorf("no cached type information for %s", ph.m.pkgPath)
	}
	data := v.(*packageData)
	return data.pkg, data.err
}

func (s *snapshot) parseGoHandles(ctx context.Context, files []span.URI, mode source.ParseMode) ([]*parseGoHandle, error) {
	pghs := make([]*parseGoHandle, 0, len(files))
	for _, uri := range files {
		fh, err := s.GetFile(ctx, uri)
		if err != nil {
			return nil, err
		}
		pghs = append(pghs, s.parseGoHandle(ctx, fh, mode))
	}
	return pghs, nil
}

func typeCheck(ctx context.Context, snapshot *snapshot, m *metadata, mode source.ParseMode, deps map[packagePath]*packageHandle) (*pkg, error) {
	var filter *unexportedFilter
	if mode == source.ParseExported {
		filter = &unexportedFilter{uses: map[string]bool{}}
	}
	pkg, err := doTypeCheck(ctx, snapshot, m, mode, deps, filter)
	if err != nil {
		return nil, err
	}

	if mode == source.ParseExported {
		// The AST filtering is a little buggy and may remove things it
		// shouldn't. If we only got undeclared name errors, try one more
		// time keeping those names.
		missing, unexpected := filter.ProcessErrors(pkg.typeErrors)
		if len(unexpected) == 0 && len(missing) != 0 {
			event.Log(ctx, fmt.Sprintf("discovered missing identifiers: %v", missing), tag.Package.Of(string(m.id)))
			pkg, err = doTypeCheck(ctx, snapshot, m, mode, deps, filter)
			if err != nil {
				return nil, err
			}
			missing, unexpected = filter.ProcessErrors(pkg.typeErrors)
		}
		if len(unexpected) != 0 || len(missing) != 0 {
			event.Log(ctx, fmt.Sprintf("falling back to safe trimming due to type errors: %v or still-missing identifiers: %v", unexpected, missing), tag.Package.Of(string(m.id)))
			pkg, err = doTypeCheck(ctx, snapshot, m, mode, deps, nil)
			if err != nil {
				return nil, err
			}
		}
	}
	// If this is a replaced module in the workspace, the version is
	// meaningless, and we don't want clients to access it.
	if m.module != nil {
		version := m.module.Version
		if source.IsWorkspaceModuleVersion(version) {
			version = ""
		}
		pkg.version = &module.Version{
			Path:    m.module.Path,
			Version: version,
		}
	}

	// We don't care about a package's errors unless we have parsed it in full.
	if mode != source.ParseFull {
		return pkg, nil
	}

	for _, e := range m.errors {
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

func doTypeCheck(ctx context.Context, snapshot *snapshot, m *metadata, mode source.ParseMode, deps map[packagePath]*packageHandle, astFilter *unexportedFilter) (*pkg, error) {
	ctx, done := event.Start(ctx, "cache.typeCheck", tag.Package.Of(string(m.id)))
	defer done()

	pkg := &pkg{
		m:       m,
		mode:    mode,
		imports: make(map[packagePath]*pkg),
		types:   types.NewPackage(string(m.pkgPath), string(m.name)),
		typesInfo: &types.Info{
			Types:      make(map[ast.Expr]types.TypeAndValue),
			Defs:       make(map[*ast.Ident]types.Object),
			Uses:       make(map[*ast.Ident]types.Object),
			Implicits:  make(map[ast.Node]types.Object),
			Selections: make(map[*ast.SelectorExpr]*types.Selection),
			Scopes:     make(map[ast.Node]*types.Scope),
		},
		typesSizes: m.typesSizes,
	}
	typeparams.InitInferred(pkg.typesInfo)

	for _, gf := range pkg.m.goFiles {
		// In the presence of line directives, we may need to report errors in
		// non-compiled Go files, so we need to register them on the package.
		// However, we only need to really parse them in ParseFull mode, when
		// the user might actually be looking at the file.
		fh, err := snapshot.GetFile(ctx, gf)
		if err != nil {
			return nil, err
		}
		goMode := source.ParseFull
		if mode != source.ParseFull {
			goMode = source.ParseHeader
		}
		pgf, err := snapshot.ParseGo(ctx, fh, goMode)
		if err != nil {
			return nil, err
		}
		pkg.goFiles = append(pkg.goFiles, pgf)
	}

	if err := parseCompiledGoFiles(ctx, snapshot, mode, pkg, astFilter); err != nil {
		return nil, err
	}

	// Use the default type information for the unsafe package.
	if m.pkgPath == "unsafe" {
		// Don't type check Unsafe: it's unnecessary, and doing so exposes a data
		// race to Unsafe.completed.
		pkg.types = types.Unsafe
		return pkg, nil
	}

	if len(m.compiledGoFiles) == 0 {
		// No files most likely means go/packages failed. Try to attach error
		// messages to the file as much as possible.
		var found bool
		for _, e := range m.errors {
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
		return nil, errors.Errorf("no parsed files for package %s, expected: %v, errors: %v", pkg.m.pkgPath, pkg.compiledGoFiles, m.errors)
	}

	cfg := &types.Config{
		Error: func(e error) {
			pkg.typeErrors = append(pkg.typeErrors, e.(types.Error))
		},
		Importer: importerFunc(func(pkgPath string) (*types.Package, error) {
			// If the context was cancelled, we should abort.
			if ctx.Err() != nil {
				return nil, ctx.Err()
			}
			dep := resolveImportPath(pkgPath, pkg, deps)
			if dep == nil {
				return nil, snapshot.missingPkgError(ctx, pkgPath)
			}
			if !source.IsValidImport(string(m.pkgPath), string(dep.m.pkgPath)) {
				return nil, errors.Errorf("invalid use of internal package %s", pkgPath)
			}
			depPkg, err := dep.check(ctx, snapshot)
			if err != nil {
				return nil, err
			}
			pkg.imports[depPkg.m.pkgPath] = depPkg
			return depPkg.types, nil
		}),
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
	_ = check.Files(files)
	// If the context was cancelled, we may have returned a ton of transient
	// errors to the type checker. Swallow them.
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}
	return pkg, nil
}

func parseCompiledGoFiles(ctx context.Context, snapshot *snapshot, mode source.ParseMode, pkg *pkg, astFilter *unexportedFilter) error {
	for _, cgf := range pkg.m.compiledGoFiles {
		fh, err := snapshot.GetFile(ctx, cgf)
		if err != nil {
			return err
		}

		var pgf *source.ParsedGoFile
		var fixed bool
		// Only parse Full through the cache -- we need to own Exported ASTs
		// to prune them.
		if mode == source.ParseFull {
			pgh := snapshot.parseGoHandle(ctx, fh, mode)
			pgf, fixed, err = snapshot.parseGo(ctx, pgh)
		} else {
			d := parseGo(ctx, snapshot.FileSet(), fh, mode)
			pgf, fixed, err = d.parsed, d.fixed, d.err
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
		pkg.hasFixedFiles = pkg.hasFixedFiles || fixed
	}
	if mode != source.ParseExported {
		return nil
	}
	if astFilter != nil {
		var files []*ast.File
		for _, cgf := range pkg.compiledGoFiles {
			files = append(files, cgf.File)
		}
		astFilter.Filter(files)
	} else {
		for _, cgf := range pkg.compiledGoFiles {
			trimAST(cgf.File)
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
		if s.isWorkspacePackage(packageID(directImporter)) {
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
			if s.isWorkspacePackage(packageID(item)) {
				break
			}

			for _, imp := range allImports[item] {
				rng, err := source.NewMappedRange(s.FileSet(), imp.cgf.Mapper, imp.imp.Pos(), imp.imp.End()).Range()
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
			m := s.getMetadata(packageID(item))
			if m == nil || m.module == nil {
				continue
			}
			modVer := module.Version{Path: m.module.Path, Version: m.module.Version}
			reference := findModuleReference(pm.File, modVer)
			if reference == nil {
				continue
			}
			rng, err := rangeFromPositions(pm.Mapper, reference.Start, reference.End)
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

// resolveImportPath resolves an import path in pkg to a package from deps.
// It should produce the same results as resolveImportPath:
// https://cs.opensource.google/go/go/+/master:src/cmd/go/internal/load/pkg.go;drc=641918ee09cb44d282a30ee8b66f99a0b63eaef9;l=990.
func resolveImportPath(importPath string, pkg *pkg, deps map[packagePath]*packageHandle) *packageHandle {
	if dep := deps[packagePath(importPath)]; dep != nil {
		return dep
	}
	// We may be in GOPATH mode, in which case we need to check vendor dirs.
	searchDir := path.Dir(pkg.PkgPath())
	for {
		vdir := packagePath(path.Join(searchDir, "vendor", importPath))
		if vdep := deps[vdir]; vdep != nil {
			return vdep
		}

		// Search until Dir doesn't take us anywhere new, e.g. "." or "/".
		next := path.Dir(searchDir)
		if searchDir == next {
			break
		}
		searchDir = next
	}

	// Vendor didn't work. Let's try minimal module compatibility mode.
	// In MMC, the packagePath is the canonical (.../vN/...) path, which
	// is hard to calculate. But the go command has already resolved the ID
	// to the non-versioned path, and we can take advantage of that.
	for _, dep := range deps {
		if dep.ID() == importPath {
			return dep
		}
	}
	return nil
}

// An importFunc is an implementation of the single-method
// types.Importer interface based on a function value.
type importerFunc func(path string) (*types.Package, error)

func (f importerFunc) Import(path string) (*types.Package, error) { return f(path) }

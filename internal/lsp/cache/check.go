// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"bytes"
	"context"
	"fmt"
	"go/ast"
	"go/scanner"
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
	m *metadata

	// key is the hashed key for the package.
	key packageHandleKey
}

func (ph *packageHandle) packageKey() packageKey {
	return packageKey{
		id:   ph.m.id,
		mode: ph.mode,
	}
}

// packageData contains the data produced by type-checking a package.
type packageData struct {
	pkg *pkg
	err error
}

// buildPackageHandle returns a packageHandle for a given package and mode.
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
		data.pkg, data.err = typeCheck(ctx, snapshot, m, mode, deps)
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
		if err != nil {
			event.Error(ctx, fmt.Sprintf("%s: no dep handle for %s", id, depID), err, tag.Snapshot.Of(s.id))
			if ctx.Err() != nil {
				return nil, nil, ctx.Err()
			}
			// One bad dependency should not prevent us from checking the entire package.
			// Add a special key to mark a bad dependency.
			depKeys = append(depKeys, packageHandleKey(fmt.Sprintf("%s import not found", id)))
			continue
		}
		deps[depHandle.m.pkgPath] = depHandle
		depKeys = append(depKeys, depHandle.key)
	}
	experimentalKey := s.View().Options().ExperimentalPackageCacheKey
	ph.key = checkPackageKey(ctx, ph.m.id, compiledGoFiles, m.config, depKeys, mode, experimentalKey)
	return ph, deps, nil
}

func (s *snapshot) workspaceParseMode(id packageID) source.ParseMode {
	if _, ws := s.isWorkspacePackage(id); ws {
		return source.ParseFull
	} else {
		return source.ParseExported
	}
}

func checkPackageKey(ctx context.Context, id packageID, pghs []*parseGoHandle, cfg *packages.Config, deps []packageHandleKey, mode source.ParseMode, experimentalKey bool) packageHandleKey {
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
	ctx, done := event.Start(ctx, "cache.importer.typeCheck", tag.Package.Of(string(m.id)))
	defer done()

	fset := snapshot.view.session.cache.fset
	pkg := &pkg{
		m:               m,
		mode:            mode,
		goFiles:         make([]*source.ParsedGoFile, len(m.goFiles)),
		compiledGoFiles: make([]*source.ParsedGoFile, len(m.compiledGoFiles)),
		imports:         make(map[packagePath]*pkg),
		typesSizes:      m.typesSizes,
		typesInfo: &types.Info{
			Types:      make(map[ast.Expr]types.TypeAndValue),
			Defs:       make(map[*ast.Ident]types.Object),
			Uses:       make(map[*ast.Ident]types.Object),
			Implicits:  make(map[ast.Node]types.Object),
			Selections: make(map[*ast.SelectorExpr]*types.Selection),
			Scopes:     make(map[ast.Node]*types.Scope),
		},
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
	var (
		files        = make([]*ast.File, len(m.compiledGoFiles))
		parseErrors  = make([]scanner.ErrorList, len(m.compiledGoFiles))
		actualErrors = make([]error, len(m.compiledGoFiles))
		wg           sync.WaitGroup

		mu             sync.Mutex
		haveFixedFiles bool
	)
	for i, cgf := range m.compiledGoFiles {
		wg.Add(1)
		go func(i int, cgf span.URI) {
			defer wg.Done()
			fh, err := snapshot.GetFile(ctx, cgf)
			if err != nil {
				actualErrors[i] = err
				return
			}
			pgh := snapshot.parseGoHandle(ctx, fh, mode)
			pgf, fixed, err := snapshot.parseGo(ctx, pgh)
			if err != nil {
				actualErrors[i] = err
				return
			}
			pkg.compiledGoFiles[i] = pgf
			files[i], parseErrors[i], actualErrors[i] = pgf.File, pgf.ParseErr, err

			// If we have fixed parse errors in any of the files, we should hide type
			// errors, as they may be completely nonsensical.
			mu.Lock()
			haveFixedFiles = haveFixedFiles || fixed
			mu.Unlock()
		}(i, cgf)
	}
	for i, gf := range m.goFiles {
		wg.Add(1)
		// We need to parse the non-compiled go files, but we don't care about their errors.
		go func(i int, gf span.URI) {
			defer wg.Done()
			fh, err := snapshot.GetFile(ctx, gf)
			if err != nil {
				return
			}
			pgf, _ := snapshot.ParseGo(ctx, fh, mode)
			pkg.goFiles[i] = pgf
		}(i, gf)
	}
	wg.Wait()
	for _, err := range actualErrors {
		if err != nil {
			return nil, err
		}
	}

	var i int
	for _, e := range parseErrors {
		if e != nil {
			parseErrors[i] = e
			i++
		}
	}
	parseErrors = parseErrors[:i]

	i = 0
	for _, f := range files {
		if f != nil {
			files[i] = f
			i++
		}
	}
	files = files[:i]

	// Use the default type information for the unsafe package.
	if pkg.m.pkgPath == "unsafe" {
		pkg.types = types.Unsafe
		// Don't type check Unsafe: it's unnecessary, and doing so exposes a data
		// race to Unsafe.completed.
		return pkg, nil
	} else if len(files) == 0 { // not the unsafe package, no parsed files
		// Try to attach error messages to the file as much as possible.
		var found bool
		for _, e := range m.errors {
			srcDiags, err := goPackagesErrorDiagnostics(ctx, snapshot, pkg, e)
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
	} else {
		pkg.types = types.NewPackage(string(m.pkgPath), string(m.name))
	}

	var typeErrors []types.Error
	cfg := &types.Config{
		Error: func(e error) {
			typeErrors = append(typeErrors, e.(types.Error))
		},
		Importer: importerFunc(func(pkgPath string) (*types.Package, error) {
			// If the context was cancelled, we should abort.
			if ctx.Err() != nil {
				return nil, ctx.Err()
			}
			dep := resolveImportPath(pkgPath, pkg, deps)
			if dep == nil {
				return nil, snapshot.missingPkgError(pkgPath)
			}
			if !isValidImport(m.pkgPath, dep.m.pkgPath) {
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
	// We want to type check cgo code if go/types supports it.
	// We passed typecheckCgo to go/packages when we Loaded.
	typesinternal.SetUsesCgo(cfg)

	check := types.NewChecker(cfg, fset, pkg.types, pkg.typesInfo)

	// Type checking errors are handled via the config, so ignore them here.
	_ = check.Files(files)
	// If the context was cancelled, we may have returned a ton of transient
	// errors to the type checker. Swallow them.
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}

	// We don't care about a package's errors unless we have parsed it in full.
	if mode != source.ParseFull {
		return pkg, nil
	}

	if len(m.errors) != 0 {
		pkg.hasListOrParseErrors = true
		for _, e := range m.errors {
			diags, err := goPackagesErrorDiagnostics(ctx, snapshot, pkg, e)
			if err != nil {
				event.Error(ctx, "unable to compute positions for list errors", err, tag.Package.Of(pkg.ID()))
				continue
			}
			pkg.diagnostics = append(pkg.diagnostics, diags...)
		}
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
	if len(parseErrors) != 0 {
		pkg.hasListOrParseErrors = true
		for _, e := range parseErrors {
			diags, err := parseErrorDiagnostics(ctx, snapshot, pkg, e)
			if err != nil {
				event.Error(ctx, "unable to compute positions for parse errors", err, tag.Package.Of(pkg.ID()))
				continue
			}
			for _, diag := range diags {
				unparseable[diag.URI] = true
				pkg.diagnostics = append(pkg.diagnostics, diag)
			}
		}
	}

	if haveFixedFiles {
		return pkg, nil
	}

	for _, e := range expandErrors(typeErrors, snapshot.View().Options().RelatedInformationSupported) {
		pkg.hasTypeErrors = true
		diags, err := typeErrorDiagnostics(ctx, snapshot, pkg, e)
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
		if _, ok := s.isWorkspacePackage(packageID(directImporter)); ok {
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
			if _, ok := s.isWorkspacePackage(packageID(item)); ok {
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
func (s *snapshot) missingPkgError(pkgPath string) error {
	if s.workspaceMode()&moduleMode != 0 {
		return fmt.Errorf("no required module provides package %q", pkgPath)
	}
	gorootSrcPkg := filepath.FromSlash(filepath.Join(s.view.goroot, "src", pkgPath))

	var b strings.Builder
	b.WriteString(fmt.Sprintf("cannot find package %q in any of \n\t%s (from $GOROOT)", pkgPath, gorootSrcPkg))

	for _, gopath := range strings.Split(s.view.gopath, ":") {
		gopathSrcPkg := filepath.FromSlash(filepath.Join(gopath, "src", pkgPath))
		b.WriteString(fmt.Sprintf("\n\t%s (from $GOPATH)", gopathSrcPkg))
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

func isValidImport(pkgPath, importPkgPath packagePath) bool {
	i := strings.LastIndex(string(importPkgPath), "/internal/")
	if i == -1 {
		return true
	}
	if pkgPath == "command-line-arguments" {
		return true
	}
	return strings.HasPrefix(string(pkgPath), string(importPkgPath[:i]))
}

// An importFunc is an implementation of the single-method
// types.Importer interface based on a function value.
type importerFunc func(path string) (*types.Package, error)

func (f importerFunc) Import(path string) (*types.Package, error) { return f(path) }

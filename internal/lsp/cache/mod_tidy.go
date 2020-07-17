// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"context"
	"fmt"
	"go/ast"
	"go/token"
	"io/ioutil"
	"sort"
	"strconv"
	"strings"

	"golang.org/x/mod/modfile"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/lsp/debug/tag"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/memoize"
	"golang.org/x/tools/internal/packagesinternal"
	"golang.org/x/tools/internal/span"
)

type modTidyKey struct {
	sessionID       string
	cfg             string
	gomod           string
	imports         string
	unsavedOverlays string
	view            string
}

type modTidyHandle struct {
	handle *memoize.Handle

	pmh source.ParseModHandle
}

type modTidyData struct {
	memoize.NoCopy

	// tidiedContent is the content of the tidied file.
	tidiedContent []byte

	// diagnostics are any errors and associated suggested fixes for
	// the go.mod file.
	diagnostics []source.Error

	err error
}

func (mth *modTidyHandle) ParseModHandle() source.ParseModHandle {
	return mth.pmh
}

func (mth *modTidyHandle) Tidy(ctx context.Context) ([]source.Error, error) {
	v, err := mth.handle.Get(ctx)
	if err != nil {
		return nil, err
	}
	data := v.(*modTidyData)
	return data.diagnostics, data.err
}

func (mth *modTidyHandle) TidiedContent(ctx context.Context) ([]byte, error) {
	v, err := mth.handle.Get(ctx)
	if err != nil {
		return nil, err
	}
	data := v.(*modTidyData)
	return data.tidiedContent, data.err
}

func (s *snapshot) ModTidyHandle(ctx context.Context) (source.ModTidyHandle, error) {
	if !s.view.tmpMod {
		return nil, source.ErrTmpModfileUnsupported
	}
	if handle := s.getModTidyHandle(); handle != nil {
		return handle, nil
	}
	fh, err := s.GetFile(ctx, s.view.modURI)
	if err != nil {
		return nil, err
	}
	pmh, err := s.ParseModHandle(ctx, fh)
	if err != nil {
		return nil, err
	}
	wsPhs, err := s.WorkspacePackages(ctx)
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}
	if err != nil {
		return nil, err
	}
	var workspacePkgs []source.Package
	for _, ph := range wsPhs {
		pkg, err := ph.Check(ctx)
		if err != nil {
			return nil, err
		}
		workspacePkgs = append(workspacePkgs, pkg)
	}
	importHash, err := hashImports(ctx, workspacePkgs)
	if err != nil {
		return nil, err
	}

	s.mu.Lock()
	overlayHash := hashUnsavedOverlays(s.files)
	s.mu.Unlock()

	var (
		modURI  = s.view.modURI
		cfg     = s.config(ctx)
		options = s.view.Options()
		fset    = s.view.session.cache.fset
	)
	key := modTidyKey{
		sessionID:       s.view.session.id,
		view:            s.view.root.Filename(),
		imports:         importHash,
		unsavedOverlays: overlayHash,
		gomod:           pmh.Mod().Identity().String(),
		cfg:             hashConfig(cfg),
	}
	h := s.view.session.cache.store.Bind(key, func(ctx context.Context) interface{} {
		ctx, done := event.Start(ctx, "cache.ModTidyHandle", tag.URI.Of(modURI))
		defer done()

		original, m, parseErrors, err := pmh.Parse(ctx)
		if err != nil || len(parseErrors) > 0 {
			return &modTidyData{
				diagnostics: parseErrors,
				err:         err,
			}
		}
		tmpURI, inv, cleanup, err := goCommandInvocation(ctx, cfg, pmh, "mod", []string{"tidy"})
		if err != nil {
			return &modTidyData{err: err}
		}
		// Keep the temporary go.mod file around long enough to parse it.
		defer cleanup()

		if _, err := packagesinternal.GetGoCmdRunner(cfg).Run(ctx, *inv); err != nil {
			return &modTidyData{err: err}
		}
		// Go directly to disk to get the temporary mod file, since it is
		// always on disk.
		tempContents, err := ioutil.ReadFile(tmpURI.Filename())
		if err != nil {
			return &modTidyData{err: err}
		}
		ideal, err := modfile.Parse(tmpURI.Filename(), tempContents, nil)
		if err != nil {
			// We do not need to worry about the temporary file's parse errors
			// since it has been "tidied".
			return &modTidyData{err: err}
		}
		// Get the dependencies that are different between the original and
		// ideal go.mod files.
		unusedDeps := make(map[string]*modfile.Require, len(original.Require))
		missingDeps := make(map[string]*modfile.Require, len(ideal.Require))
		for _, req := range original.Require {
			unusedDeps[req.Mod.Path] = req
		}
		for _, req := range ideal.Require {
			origDep := unusedDeps[req.Mod.Path]
			if origDep != nil && origDep.Indirect == req.Indirect {
				delete(unusedDeps, req.Mod.Path)
			} else {
				missingDeps[req.Mod.Path] = req
			}
		}
		// First, compute any errors specific to the go.mod file. These include
		// unused dependencies and modules with incorrect // indirect comments.
		/// Both the diagnostic and the fix will appear on the go.mod file.
		modRequireErrs, err := modRequireErrors(m, missingDeps, unusedDeps, options)
		if err != nil {
			return &modTidyData{err: err}
		}
		for _, req := range missingDeps {
			if unusedDeps[req.Mod.Path] != nil {
				delete(missingDeps, req.Mod.Path)
			}
		}
		// Next, compute any diagnostics for modules that are missing from the
		// go.mod file. The fixes will be for the go.mod file, but the
		// diagnostics should appear on the import statements in the Go or
		// go.mod files.
		missingModuleErrs, err := missingModuleErrors(ctx, fset, m, workspacePkgs, ideal.Require, missingDeps, original, options)
		if err != nil {
			return &modTidyData{err: err}
		}
		return &modTidyData{
			tidiedContent: tempContents,
			diagnostics:   append(modRequireErrs, missingModuleErrs...),
		}
	})
	s.mu.Lock()
	defer s.mu.Unlock()
	s.modTidyHandle = &modTidyHandle{
		handle: h,
		pmh:    pmh,
	}
	return s.modTidyHandle, nil
}

func hashImports(ctx context.Context, wsPackages []source.Package) (string, error) {
	results := make(map[string]bool)
	var imports []string
	for _, pkg := range wsPackages {
		for _, path := range pkg.Imports() {
			imp := path.PkgPath()
			if _, ok := results[imp]; !ok {
				results[imp] = true
				imports = append(imports, imp)
			}
		}
	}
	sort.Strings(imports)
	hashed := strings.Join(imports, ",")
	return hashContents([]byte(hashed)), nil
}

// modRequireErrors extracts the errors that occur on the require directives.
// It checks for directness issues and unused dependencies.
func modRequireErrors(m *protocol.ColumnMapper, missingDeps, unusedDeps map[string]*modfile.Require, options source.Options) ([]source.Error, error) {
	var errors []source.Error
	for dep, req := range unusedDeps {
		if req.Syntax == nil {
			continue
		}
		// Handle dependencies that are incorrectly labeled indirect and vice versa.
		if missingDeps[dep] != nil && req.Indirect != missingDeps[dep].Indirect {
			directErr, err := modDirectnessErrors(m, req, options)
			if err != nil {
				return nil, err
			}
			errors = append(errors, directErr)
		}
		// Handle unused dependencies.
		if missingDeps[dep] == nil {
			rng, err := rangeFromPositions(m, req.Syntax.Start, req.Syntax.End)
			if err != nil {
				return nil, err
			}
			edits, err := dropDependencyEdits(m, req, options)
			if err != nil {
				return nil, err
			}
			errors = append(errors, source.Error{
				Category: ModTidyError,
				Message:  fmt.Sprintf("%s is not used in this module.", dep),
				Range:    rng,
				URI:      m.URI,
				SuggestedFixes: []source.SuggestedFix{{
					Title: fmt.Sprintf("Remove dependency: %s", dep),
					Edits: map[span.URI][]protocol.TextEdit{
						m.URI: edits,
					},
				}},
			})
		}
	}
	return errors, nil
}

const ModTidyError = "go mod tidy"

// modDirectnessErrors extracts errors when a dependency is labeled indirect when it should be direct and vice versa.
func modDirectnessErrors(m *protocol.ColumnMapper, req *modfile.Require, options source.Options) (source.Error, error) {
	rng, err := rangeFromPositions(m, req.Syntax.Start, req.Syntax.End)
	if err != nil {
		return source.Error{}, err
	}
	if req.Indirect {
		// If the dependency should be direct, just highlight the // indirect.
		if comments := req.Syntax.Comment(); comments != nil && len(comments.Suffix) > 0 {
			end := comments.Suffix[0].Start
			end.LineRune += len(comments.Suffix[0].Token)
			end.Byte += len([]byte(comments.Suffix[0].Token))
			rng, err = rangeFromPositions(m, comments.Suffix[0].Start, end)
			if err != nil {
				return source.Error{}, err
			}
		}
		edits, err := changeDirectnessEdits(m, req, false, options)
		if err != nil {
			return source.Error{}, err
		}
		return source.Error{
			Category: ModTidyError,
			Message:  fmt.Sprintf("%s should be a direct dependency", req.Mod.Path),
			Range:    rng,
			URI:      m.URI,
			SuggestedFixes: []source.SuggestedFix{{
				Title: fmt.Sprintf("Make %s direct", req.Mod.Path),
				Edits: map[span.URI][]protocol.TextEdit{
					m.URI: edits,
				},
			}},
		}, nil
	}
	// If the dependency should be indirect, add the // indirect.
	edits, err := changeDirectnessEdits(m, req, true, options)
	if err != nil {
		return source.Error{}, err
	}
	return source.Error{
		Category: ModTidyError,
		Message:  fmt.Sprintf("%s should be an indirect dependency", req.Mod.Path),
		Range:    rng,
		URI:      m.URI,
		SuggestedFixes: []source.SuggestedFix{{
			Title: fmt.Sprintf("Make %s indirect", req.Mod.Path),
			Edits: map[span.URI][]protocol.TextEdit{
				m.URI: edits,
			},
		}},
	}, nil
}

// dropDependencyEdits gets the edits needed to remove the dependency from the go.mod file.
// As an example, this function will codify the edits needed to convert the before go.mod file to the after.
// Before:
// 	module t
//
// 	go 1.11
//
// 	require golang.org/x/mod v0.1.1-0.20191105210325-c90efee705ee
// After:
// 	module t
//
// 	go 1.11
func dropDependencyEdits(m *protocol.ColumnMapper, req *modfile.Require, options source.Options) ([]protocol.TextEdit, error) {
	// We need a private copy of the parsed go.mod file, since we're going to
	// modify it.
	copied, err := modfile.Parse("", m.Content, nil)
	if err != nil {
		return nil, err
	}
	if err := copied.DropRequire(req.Mod.Path); err != nil {
		return nil, err
	}
	copied.Cleanup()
	newContent, err := copied.Format()
	if err != nil {
		return nil, err
	}
	// Calculate the edits to be made due to the change.
	diff := options.ComputeEdits(m.URI, string(m.Content), string(newContent))
	edits, err := source.ToProtocolEdits(m, diff)
	if err != nil {
		return nil, err
	}
	return edits, nil
}

// changeDirectnessEdits gets the edits needed to change an indirect dependency to direct and vice versa.
// As an example, this function will codify the edits needed to convert the before go.mod file to the after.
// Before:
// 	module t
//
// 	go 1.11
//
// 	require golang.org/x/mod v0.1.1-0.20191105210325-c90efee705ee
// After:
// 	module t
//
// 	go 1.11
//
// 	require golang.org/x/mod v0.1.1-0.20191105210325-c90efee705ee // indirect
func changeDirectnessEdits(m *protocol.ColumnMapper, req *modfile.Require, indirect bool, options source.Options) ([]protocol.TextEdit, error) {
	// We need a private copy of the parsed go.mod file, since we're going to
	// modify it.
	copied, err := modfile.Parse("", m.Content, nil)
	if err != nil {
		return nil, err
	}
	// Change the directness in the matching require statement. To avoid
	// reordering the require statements, rewrite all of them.
	var requires []*modfile.Require
	for _, r := range copied.Require {
		if r.Mod.Path == req.Mod.Path {
			requires = append(requires, &modfile.Require{
				Mod:      r.Mod,
				Syntax:   r.Syntax,
				Indirect: indirect,
			})
			continue
		}
		requires = append(requires, r)
	}
	copied.SetRequire(requires)
	newContent, err := copied.Format()
	if err != nil {
		return nil, err
	}
	// Calculate the edits to be made due to the change.
	diff := options.ComputeEdits(m.URI, string(m.Content), string(newContent))
	edits, err := source.ToProtocolEdits(m, diff)
	if err != nil {
		return nil, err
	}
	return edits, nil
}

func rangeFromPositions(m *protocol.ColumnMapper, s, e modfile.Position) (protocol.Range, error) {
	toPoint := func(offset int) (span.Point, error) {
		l, c, err := m.Converter.ToPosition(offset)
		if err != nil {
			return span.Point{}, err
		}
		return span.NewPoint(l, c, offset), nil
	}
	start, err := toPoint(s.Byte)
	if err != nil {
		return protocol.Range{}, err
	}
	end, err := toPoint(e.Byte)
	if err != nil {
		return protocol.Range{}, err
	}
	return m.Range(span.New(m.URI, start, end))
}

// missingModuleErrors returns diagnostics for each file in each workspace
// package that has dependencies that are not reflected in the go.mod file.
func missingModuleErrors(ctx context.Context, fset *token.FileSet, modMapper *protocol.ColumnMapper, pkgs []source.Package, modules []*modfile.Require, missingMods map[string]*modfile.Require, original *modfile.File, options source.Options) ([]source.Error, error) {
	var moduleErrs []source.Error
	matchedMissingMods := make(map[*modfile.Require]struct{})
	for _, pkg := range pkgs {
		missingPkgs := map[string]*modfile.Require{}
		for _, imp := range pkg.Imports() {
			if req, ok := missingMods[imp.PkgPath()]; ok {
				missingPkgs[imp.PkgPath()] = req
				break
			}
			// If the import is a package of the dependency, then add the
			// package to the map, this will eliminate the need to do this
			// prefix package search on each import for each file.
			// Example:
			//
			// import (
			//   "golang.org/x/tools/go/expect"
			//   "golang.org/x/tools/go/packages"
			// )
			// They both are related to the same module: "golang.org/x/tools".
			var match string
			for _, mod := range modules {
				if strings.HasPrefix(imp.PkgPath(), mod.Mod.Path) && len(mod.Mod.Path) > len(match) {
					match = mod.Mod.Path
				}
			}
			if req, ok := missingMods[match]; ok {
				missingPkgs[imp.PkgPath()] = req
				matchedMissingMods[req] = struct{}{}
			}
		}
		if len(missingPkgs) > 0 {
			errs, err := missingModules(ctx, fset, modMapper, pkg, missingPkgs, options)
			if err != nil {
				return nil, err
			}
			moduleErrs = append(moduleErrs, errs...)
		}
	}
	for _, req := range missingMods {
		if _, ok := matchedMissingMods[req]; ok {
			continue
		}
		s, e := original.Module.Syntax.Span()
		rng, err := rangeFromPositions(modMapper, s, e)
		if err != nil {
			return nil, err
		}
		edits, err := addRequireFix(modMapper, req, options)
		if err != nil {
			return nil, err
		}
		moduleErrs = append(moduleErrs, source.Error{
			URI:      modMapper.URI,
			Range:    rng,
			Message:  fmt.Sprintf("%s is not in your go.mod file", req.Mod.Path),
			Category: "go mod tidy",
			Kind:     source.ModTidyError,
			SuggestedFixes: []source.SuggestedFix{
				{
					Title: "Add %s to your go.mod file",
					Edits: edits,
				},
			},
		})
	}
	return moduleErrs, nil
}

func missingModules(ctx context.Context, fset *token.FileSet, modMapper *protocol.ColumnMapper, pkg source.Package, missing map[string]*modfile.Require, options source.Options) ([]source.Error, error) {
	var errors []source.Error
	for _, pgh := range pkg.CompiledGoFiles() {
		file, _, m, _, err := pgh.Parse(ctx)
		if err != nil {
			return nil, err
		}
		imports := make(map[string]*ast.ImportSpec)
		for _, imp := range file.Imports {
			if imp.Path == nil {
				continue
			}
			if target, err := strconv.Unquote(imp.Path.Value); err == nil {
				imports[target] = imp
			}
		}
		if len(imports) == 0 {
			continue
		}
		for mod, req := range missing {
			if req.Syntax == nil {
				continue
			}
			imp, ok := imports[mod]
			if !ok {
				continue
			}
			spn, err := span.NewRange(fset, imp.Path.Pos(), imp.Path.End()).Span()
			if err != nil {
				return nil, err
			}
			rng, err := m.Range(spn)
			if err != nil {
				return nil, err
			}
			edits, err := addRequireFix(modMapper, req, options)
			if err != nil {
				return nil, err
			}
			errors = append(errors, source.Error{
				URI:      pgh.File().URI(),
				Range:    rng,
				Message:  fmt.Sprintf("%s is not in your go.mod file", req.Mod.Path),
				Category: "go mod tidy",
				Kind:     source.ModTidyError,
				SuggestedFixes: []source.SuggestedFix{
					{
						Title: "Add %s to your go.mod file",
						Edits: edits,
					},
				},
			})
		}
	}
	return errors, nil
}

// addRequireFix creates edits for adding a given require to a go.mod file.
func addRequireFix(m *protocol.ColumnMapper, req *modfile.Require, options source.Options) (map[span.URI][]protocol.TextEdit, error) {
	// We need a private copy of the parsed go.mod file, since we're going to
	// modify it.
	copied, err := modfile.Parse("", m.Content, nil)
	if err != nil {
		return nil, err
	}
	// Calculate the quick fix edits that need to be made to the go.mod file.
	if err := copied.AddRequire(req.Mod.Path, req.Mod.Version); err != nil {
		return nil, err
	}
	copied.SortBlocks()
	newContents, err := copied.Format()
	if err != nil {
		return nil, err
	}
	// Calculate the edits to be made due to the change.
	diff := options.ComputeEdits(m.URI, string(m.Content), string(newContents))
	edits, err := source.ToProtocolEdits(m, diff)
	if err != nil {
		return nil, err
	}
	return map[span.URI][]protocol.TextEdit{
		m.URI: edits,
	}, nil
}

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"context"
	"fmt"
	"go/ast"
	"io/ioutil"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"golang.org/x/mod/modfile"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/gocommand"
	"golang.org/x/tools/internal/lsp/debug/tag"
	"golang.org/x/tools/internal/lsp/diff"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/memoize"
	"golang.org/x/tools/internal/span"
)

type modTidyKey struct {
	sessionID       string
	env             string
	gomod           source.FileIdentity
	imports         string
	unsavedOverlays string
	view            string
}

type modTidyHandle struct {
	handle *memoize.Handle
}

type modTidyData struct {
	tidied *source.TidiedModule
	err    error
}

func (mth *modTidyHandle) tidy(ctx context.Context, snapshot *snapshot) (*source.TidiedModule, error) {
	v, err := mth.handle.Get(ctx, snapshot.generation, snapshot)
	if err != nil {
		return nil, err
	}
	data := v.(*modTidyData)
	return data.tidied, data.err
}

func (s *snapshot) ModTidy(ctx context.Context, pm *source.ParsedModule) (*source.TidiedModule, error) {
	if pm.File == nil {
		return nil, fmt.Errorf("cannot tidy unparseable go.mod file: %v", pm.URI)
	}
	if handle := s.getModTidyHandle(pm.URI); handle != nil {
		return handle.tidy(ctx, s)
	}
	fh, err := s.GetFile(ctx, pm.URI)
	if err != nil {
		return nil, err
	}
	// If the file handle is an overlay, it may not be written to disk.
	// The go.mod file has to be on disk for `go mod tidy` to work.
	if _, ok := fh.(*overlay); ok {
		if info, _ := os.Stat(fh.URI().Filename()); info == nil {
			return nil, source.ErrNoModOnDisk
		}
	}
	if criticalErr := s.GetCriticalError(ctx); criticalErr != nil {
		return &source.TidiedModule{
			Errors: criticalErr.ErrorList,
		}, nil
	}
	workspacePkgs, err := s.WorkspacePackages(ctx)
	if err != nil {
		return nil, err
	}
	importHash, err := hashImports(ctx, workspacePkgs)
	if err != nil {
		return nil, err
	}

	s.mu.Lock()
	overlayHash := hashUnsavedOverlays(s.files)
	s.mu.Unlock()

	key := modTidyKey{
		sessionID:       s.view.session.id,
		view:            s.view.folder.Filename(),
		imports:         importHash,
		unsavedOverlays: overlayHash,
		gomod:           fh.FileIdentity(),
		env:             hashEnv(s),
	}
	h := s.generation.Bind(key, func(ctx context.Context, arg memoize.Arg) interface{} {
		ctx, done := event.Start(ctx, "cache.ModTidyHandle", tag.URI.Of(fh.URI()))
		defer done()

		snapshot := arg.(*snapshot)
		inv := &gocommand.Invocation{
			Verb:       "mod",
			Args:       []string{"tidy"},
			WorkingDir: filepath.Dir(fh.URI().Filename()),
		}
		tmpURI, inv, cleanup, err := snapshot.goCommandInvocation(ctx, source.WriteTemporaryModFile|source.AllowNetwork, inv)
		if err != nil {
			return &modTidyData{err: err}
		}
		// Keep the temporary go.mod file around long enough to parse it.
		defer cleanup()

		if _, err := s.view.session.gocmdRunner.Run(ctx, *inv); err != nil {
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
		// Compare the original and tidied go.mod files to compute errors and
		// suggested fixes.
		errors, err := modTidyErrors(ctx, snapshot, pm, ideal, workspacePkgs)
		if err != nil {
			return &modTidyData{err: err}
		}
		return &modTidyData{
			tidied: &source.TidiedModule{
				Errors:        errors,
				TidiedContent: tempContents,
			},
		}
	}, nil)

	mth := &modTidyHandle{handle: h}
	s.mu.Lock()
	s.modTidyHandles[fh.URI()] = mth
	s.mu.Unlock()

	return mth.tidy(ctx, s)
}

func (s *snapshot) parseModError(ctx context.Context, fh source.FileHandle, errText string) *source.Error {
	// Match on common error messages. This is really hacky, but I'm not sure
	// of any better way. This can be removed when golang/go#39164 is resolved.
	isInconsistentVendor := strings.Contains(errText, "inconsistent vendoring")
	isGoSumUpdates := strings.Contains(errText, "updates to go.sum needed") || strings.Contains(errText, "missing go.sum entry")

	if !isInconsistentVendor && !isGoSumUpdates {
		return nil
	}

	pmf, err := s.ParseMod(ctx, fh)
	if err != nil {
		return nil
	}
	if pmf.File.Module == nil || pmf.File.Module.Syntax == nil {
		return nil
	}
	rng, err := rangeFromPositions(pmf.Mapper, pmf.File.Module.Syntax.Start, pmf.File.Module.Syntax.End)
	if err != nil {
		return nil
	}
	args, err := source.MarshalArgs(protocol.URIFromSpanURI(fh.URI()))
	if err != nil {
		return nil
	}

	switch {
	case isInconsistentVendor:
		return &source.Error{
			URI:   fh.URI(),
			Range: rng,
			Kind:  source.ListError,
			Message: `Inconsistent vendoring detected. Please re-run "go mod vendor".
See https://github.com/golang/go/issues/39164 for more detail on this issue.`,
			SuggestedFixes: []source.SuggestedFix{{
				Title: source.CommandVendor.Title,
				Command: &protocol.Command{
					Command:   source.CommandVendor.ID(),
					Title:     source.CommandVendor.Title,
					Arguments: args,
				},
			}},
		}

	case isGoSumUpdates:
		return &source.Error{
			URI:     fh.URI(),
			Range:   rng,
			Kind:    source.ListError,
			Message: `go.sum is out of sync with go.mod. Please update it or run "go mod tidy".`,
			SuggestedFixes: []source.SuggestedFix{
				{
					Title: source.CommandTidy.Title,
					Command: &protocol.Command{
						Command:   source.CommandTidy.ID(),
						Title:     source.CommandTidy.Title,
						Arguments: args,
					},
				},
				{
					Title: source.CommandUpdateGoSum.Title,
					Command: &protocol.Command{
						Command:   source.CommandUpdateGoSum.ID(),
						Title:     source.CommandUpdateGoSum.Title,
						Arguments: args,
					},
				},
			},
		}
	}
	return nil
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
		imports = append(imports, pkg.MissingDependencies()...)
	}
	sort.Strings(imports)
	hashed := strings.Join(imports, ",")
	return hashContents([]byte(hashed)), nil
}

// modTidyErrors computes the differences between the original and tidied
// go.mod files to produce diagnostic and suggested fixes. Some diagnostics
// may appear on the Go files that import packages from missing modules.
func modTidyErrors(ctx context.Context, snapshot source.Snapshot, pm *source.ParsedModule, ideal *modfile.File, workspacePkgs []source.Package) (errors []*source.Error, err error) {
	// First, determine which modules are unused and which are missing from the
	// original go.mod file.
	var (
		unused          = make(map[string]*modfile.Require, len(pm.File.Require))
		missing         = make(map[string]*modfile.Require, len(ideal.Require))
		wrongDirectness = make(map[string]*modfile.Require, len(pm.File.Require))
	)
	for _, req := range pm.File.Require {
		unused[req.Mod.Path] = req
	}
	for _, req := range ideal.Require {
		origReq := unused[req.Mod.Path]
		if origReq == nil {
			missing[req.Mod.Path] = req
			continue
		} else if origReq.Indirect != req.Indirect {
			wrongDirectness[req.Mod.Path] = origReq
		}
		delete(unused, req.Mod.Path)
	}
	for _, req := range wrongDirectness {
		// Handle dependencies that are incorrectly labeled indirect and
		// vice versa.
		srcErr, err := directnessError(pm.Mapper, req, snapshot.View().Options().ComputeEdits)
		if err != nil {
			return nil, err
		}
		errors = append(errors, srcErr)
	}
	// Next, compute any diagnostics for modules that are missing from the
	// go.mod file. The fixes will be for the go.mod file, but the
	// diagnostics should also appear in both the go.mod file and the import
	// statements in the Go files in which the dependencies are used.
	missingModuleFixes := map[*modfile.Require][]source.SuggestedFix{}
	for _, req := range missing {
		srcErr, err := missingModuleError(snapshot, pm, req)
		if err != nil {
			return nil, err
		}
		missingModuleFixes[req] = srcErr.SuggestedFixes
		errors = append(errors, srcErr)
	}
	// Add diagnostics for missing modules anywhere they are imported in the
	// workspace.
	for _, pkg := range workspacePkgs {
		missingImports := map[string]*modfile.Require{}
		var importedPkgs []string

		// If -mod=readonly is not set we may have successfully imported
		// packages from missing modules. Otherwise they'll be in
		// MissingDependencies. Combine both.
		for _, imp := range pkg.Imports() {
			importedPkgs = append(importedPkgs, imp.PkgPath())
		}
		importedPkgs = append(importedPkgs, pkg.MissingDependencies()...)

		for _, imp := range importedPkgs {
			if req, ok := missing[imp]; ok {
				missingImports[imp] = req
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
			for _, req := range ideal.Require {
				if strings.HasPrefix(imp, req.Mod.Path) && len(req.Mod.Path) > len(match) {
					match = req.Mod.Path
				}
			}
			if req, ok := missing[match]; ok {
				missingImports[imp] = req
			}
		}
		// None of this package's imports are from missing modules.
		if len(missingImports) == 0 {
			continue
		}
		for _, pgf := range pkg.CompiledGoFiles() {
			file, m := pgf.File, pgf.Mapper
			if file == nil || m == nil {
				continue
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
			for importPath, req := range missingImports {
				imp, ok := imports[importPath]
				if !ok {
					continue
				}
				fixes, ok := missingModuleFixes[req]
				if !ok {
					return nil, fmt.Errorf("no missing module fix for %q (%q)", importPath, req.Mod.Path)
				}
				srcErr, err := missingModuleForImport(snapshot, m, imp, req, fixes)
				if err != nil {
					return nil, err
				}
				errors = append(errors, srcErr)
			}
		}
	}
	// Finally, add errors for any unused dependencies.
	onlyError := len(errors) == 0 && len(unused) == 1
	for _, req := range unused {
		srcErr, err := unusedError(pm.Mapper, req, onlyError, snapshot.View().Options().ComputeEdits)
		if err != nil {
			return nil, err
		}
		errors = append(errors, srcErr)
	}
	return errors, nil
}

// unusedError returns a source.Error for an unused require.
func unusedError(m *protocol.ColumnMapper, req *modfile.Require, onlyError bool, computeEdits diff.ComputeEdits) (*source.Error, error) {
	rng, err := rangeFromPositions(m, req.Syntax.Start, req.Syntax.End)
	if err != nil {
		return nil, err
	}
	args, err := source.MarshalArgs(m.URI, onlyError, req.Mod.Path)
	if err != nil {
		return nil, err
	}
	return &source.Error{
		Category: source.GoModTidy,
		Message:  fmt.Sprintf("%s is not used in this module", req.Mod.Path),
		Range:    rng,
		URI:      m.URI,
		SuggestedFixes: []source.SuggestedFix{{
			Title: fmt.Sprintf("Remove dependency: %s", req.Mod.Path),
			Command: &protocol.Command{
				Title:     source.CommandRemoveDependency.Title,
				Command:   source.CommandRemoveDependency.ID(),
				Arguments: args,
			},
		}},
	}, nil
}

// directnessError extracts errors when a dependency is labeled indirect when
// it should be direct and vice versa.
func directnessError(m *protocol.ColumnMapper, req *modfile.Require, computeEdits diff.ComputeEdits) (*source.Error, error) {
	rng, err := rangeFromPositions(m, req.Syntax.Start, req.Syntax.End)
	if err != nil {
		return nil, err
	}
	direction := "indirect"
	if req.Indirect {
		direction = "direct"

		// If the dependency should be direct, just highlight the // indirect.
		if comments := req.Syntax.Comment(); comments != nil && len(comments.Suffix) > 0 {
			end := comments.Suffix[0].Start
			end.LineRune += len(comments.Suffix[0].Token)
			end.Byte += len([]byte(comments.Suffix[0].Token))
			rng, err = rangeFromPositions(m, comments.Suffix[0].Start, end)
			if err != nil {
				return nil, err
			}
		}
	}
	// If the dependency should be indirect, add the // indirect.
	edits, err := switchDirectness(req, m, computeEdits)
	if err != nil {
		return nil, err
	}
	return &source.Error{
		Message:  fmt.Sprintf("%s should be %s", req.Mod.Path, direction),
		Range:    rng,
		URI:      m.URI,
		Category: source.GoModTidy,
		SuggestedFixes: []source.SuggestedFix{{
			Title: fmt.Sprintf("Change %s to %s", req.Mod.Path, direction),
			Edits: map[span.URI][]protocol.TextEdit{
				m.URI: edits,
			},
		}},
	}, nil
}

func missingModuleError(snapshot source.Snapshot, pm *source.ParsedModule, req *modfile.Require) (*source.Error, error) {
	var rng protocol.Range
	// Default to the start of the file if there is no module declaration.
	if pm.File != nil && pm.File.Module != nil && pm.File.Module.Syntax != nil {
		start, end := pm.File.Module.Syntax.Span()
		var err error
		rng, err = rangeFromPositions(pm.Mapper, start, end)
		if err != nil {
			return nil, err
		}
	}
	args, err := source.MarshalArgs(pm.Mapper.URI, !req.Indirect, []string{req.Mod.Path + "@" + req.Mod.Version})
	if err != nil {
		return nil, err
	}
	return &source.Error{
		URI:      pm.Mapper.URI,
		Range:    rng,
		Message:  fmt.Sprintf("%s is not in your go.mod file", req.Mod.Path),
		Category: source.GoModTidy,
		Kind:     source.ModTidyError,
		SuggestedFixes: []source.SuggestedFix{{
			Title: fmt.Sprintf("Add %s to your go.mod file", req.Mod.Path),
			Command: &protocol.Command{
				Title:     source.CommandAddDependency.Title,
				Command:   source.CommandAddDependency.ID(),
				Arguments: args,
			},
		}},
	}, nil
}

// switchDirectness gets the edits needed to change an indirect dependency to
// direct and vice versa.
func switchDirectness(req *modfile.Require, m *protocol.ColumnMapper, computeEdits diff.ComputeEdits) ([]protocol.TextEdit, error) {
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
				Indirect: !r.Indirect,
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
	diff, err := computeEdits(m.URI, string(m.Content), string(newContent))
	if err != nil {
		return nil, err
	}
	return source.ToProtocolEdits(m, diff)
}

// missingModuleForImport creates an error for a given import path that comes
// from a missing module.
func missingModuleForImport(snapshot source.Snapshot, m *protocol.ColumnMapper, imp *ast.ImportSpec, req *modfile.Require, fixes []source.SuggestedFix) (*source.Error, error) {
	if req.Syntax == nil {
		return nil, fmt.Errorf("no syntax for %v", req)
	}
	spn, err := span.NewRange(snapshot.FileSet(), imp.Path.Pos(), imp.Path.End()).Span()
	if err != nil {
		return nil, err
	}
	rng, err := m.Range(spn)
	if err != nil {
		return nil, err
	}
	return &source.Error{
		Category:       source.GoModTidy,
		URI:            m.URI,
		Range:          rng,
		Message:        fmt.Sprintf("%s is not in your go.mod file", req.Mod.Path),
		Kind:           source.ModTidyError,
		SuggestedFixes: fixes,
	}, nil
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

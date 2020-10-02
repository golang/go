// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"
	"fmt"
	"go/ast"
	"go/token"
	"go/types"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/internal/lsp/analysis/fillstruct"
	"golang.org/x/tools/internal/lsp/analysis/undeclaredname"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/span"
	errors "golang.org/x/xerrors"
)

type Command struct {
	Title string
	Name  string

	// Synchronous controls whether the command executes synchronously within the
	// ExecuteCommand request (applying suggested fixes is always synchronous).
	Synchronous bool

	// appliesFn is an optional field to indicate whether or not a command can
	// be applied to the given inputs. If it returns false, we should not
	// suggest this command for these inputs.
	appliesFn AppliesFunc

	// suggestedFixFn is an optional field to generate the edits that the
	// command produces for the given inputs.
	suggestedFixFn SuggestedFixFunc
}

// ID adds the "gopls_" prefix to the command name, in order to avoid
// collisions with other language servers.
func (c Command) ID() string {
	return "gopls_" + c.Name
}

type AppliesFunc func(fset *token.FileSet, rng span.Range, src []byte, file *ast.File, pkg *types.Package, info *types.Info) bool

// SuggestedFixFunc is a function used to get the suggested fixes for a given
// gopls command, some of which are provided by go/analysis.Analyzers. Some of
// the analyzers in internal/lsp/analysis are not efficient enough to include
// suggested fixes with their diagnostics, so we have to compute them
// separately. Such analyzers should provide a function with a signature of
// SuggestedFixFunc.
type SuggestedFixFunc func(fset *token.FileSet, rng span.Range, src []byte, file *ast.File, pkg *types.Package, info *types.Info) (*analysis.SuggestedFix, error)

// Commands are the commands currently supported by gopls.
var Commands = []*Command{
	CommandGenerate,
	CommandFillStruct,
	CommandRegenerateCgo,
	CommandTest,
	CommandTidy,
	CommandUndeclaredName,
	CommandUpgradeDependency,
	CommandVendor,
	CommandExtractVariable,
	CommandExtractFunction,
	CommandToggleDetails,
	CommandGenerateGoplsMod,
}

var (
	// CommandTest runs `go test` for a specific test function.
	CommandTest = &Command{
		Name:  "test",
		Title: "Run test(s)",
	}

	// CommandGenerate runs `go generate` for a given directory.
	CommandGenerate = &Command{
		Name:  "generate",
		Title: "Run go generate",
	}

	// CommandTidy runs `go mod tidy` for a module.
	CommandTidy = &Command{
		Name:  "tidy",
		Title: "Run go mod tidy",
	}

	// CommandVendor runs `go mod vendor` for a module.
	CommandVendor = &Command{
		Name:  "vendor",
		Title: "Run go mod vendor",
	}

	// CommandUpgradeDependency upgrades a dependency.
	CommandUpgradeDependency = &Command{
		Name:  "upgrade_dependency",
		Title: "Upgrade dependency",
	}

	// CommandRegenerateCgo regenerates cgo definitions.
	CommandRegenerateCgo = &Command{
		Name:  "regenerate_cgo",
		Title: "Regenerate cgo",
	}

	// CommandToggleDetails controls calculation of gc annotations.
	CommandToggleDetails = &Command{
		Name:  "gc_details",
		Title: "Toggle gc_details",
	}

	// CommandFillStruct is a gopls command to fill a struct with default
	// values.
	CommandFillStruct = &Command{
		Name:           "fill_struct",
		Title:          "Fill struct",
		suggestedFixFn: fillstruct.SuggestedFix,
	}

	// CommandUndeclaredName adds a variable declaration for an undeclared
	// name.
	CommandUndeclaredName = &Command{
		Name:           "undeclared_name",
		Title:          "Undeclared name",
		suggestedFixFn: undeclaredname.SuggestedFix,
	}

	// CommandExtractVariable extracts an expression to a variable.
	CommandExtractVariable = &Command{
		Name:           "extract_variable",
		Title:          "Extract to variable",
		suggestedFixFn: extractVariable,
		appliesFn: func(_ *token.FileSet, rng span.Range, _ []byte, file *ast.File, _ *types.Package, _ *types.Info) bool {
			_, _, ok, _ := canExtractVariable(rng, file)
			return ok
		},
	}

	// CommandExtractFunction extracts statements to a function.
	CommandExtractFunction = &Command{
		Name:           "extract_function",
		Title:          "Extract to function",
		suggestedFixFn: extractFunction,
		appliesFn: func(fset *token.FileSet, rng span.Range, src []byte, file *ast.File, _ *types.Package, info *types.Info) bool {
			_, ok, _ := canExtractFunction(fset, rng, src, file, info)
			return ok
		},
	}

	// CommandGenerateGoplsMod (re)generates the gopls.mod file.
	CommandGenerateGoplsMod = &Command{
		Name:        "generate_gopls_mod",
		Title:       "Generate gopls.mod",
		Synchronous: true,
	}
)

// Applies reports whether the command c implements a suggested fix that is
// relevant to the given rng.
func (c *Command) Applies(ctx context.Context, snapshot Snapshot, fh FileHandle, pRng protocol.Range) bool {
	// If there is no applies function, assume that the command applies.
	if c.appliesFn == nil {
		return true
	}
	fset, rng, src, file, _, pkg, info, err := getAllSuggestedFixInputs(ctx, snapshot, fh, pRng)
	if err != nil {
		return false
	}
	return c.appliesFn(fset, rng, src, file, pkg, info)
}

// IsSuggestedFix reports whether the given command is intended to work as a
// suggested fix. Suggested fix commands are intended to return edits which are
// then applied to the workspace.
func (c *Command) IsSuggestedFix() bool {
	return c.suggestedFixFn != nil
}

// SuggestedFix applies the command's suggested fix to the given file and
// range, returning the resulting edits.
func (c *Command) SuggestedFix(ctx context.Context, snapshot Snapshot, fh VersionedFileHandle, pRng protocol.Range) ([]protocol.TextDocumentEdit, error) {
	if c.suggestedFixFn == nil {
		return nil, fmt.Errorf("no suggested fix function for %s", c.Name)
	}
	fset, rng, src, file, m, pkg, info, err := getAllSuggestedFixInputs(ctx, snapshot, fh, pRng)
	if err != nil {
		return nil, err
	}
	fix, err := c.suggestedFixFn(fset, rng, src, file, pkg, info)
	if err != nil {
		return nil, err
	}
	if fix == nil {
		return nil, nil
	}

	var edits []protocol.TextDocumentEdit
	for _, edit := range fix.TextEdits {
		rng := span.NewRange(fset, edit.Pos, edit.End)
		spn, err := rng.Span()
		if err != nil {
			return nil, err
		}
		clRng, err := m.Range(spn)
		if err != nil {
			return nil, err
		}
		edits = append(edits, protocol.TextDocumentEdit{
			TextDocument: protocol.VersionedTextDocumentIdentifier{
				Version: fh.Version(),
				TextDocumentIdentifier: protocol.TextDocumentIdentifier{
					URI: protocol.URIFromSpanURI(fh.URI()),
				},
			},
			Edits: []protocol.TextEdit{
				{
					Range:   clRng,
					NewText: string(edit.NewText),
				},
			},
		})
	}
	return edits, nil
}

// getAllSuggestedFixInputs is a helper function to collect all possible needed
// inputs for an AppliesFunc or SuggestedFixFunc.
func getAllSuggestedFixInputs(ctx context.Context, snapshot Snapshot, fh FileHandle, pRng protocol.Range) (*token.FileSet, span.Range, []byte, *ast.File, *protocol.ColumnMapper, *types.Package, *types.Info, error) {
	pkg, pgf, err := GetParsedFile(ctx, snapshot, fh, NarrowestPackage)
	if err != nil {
		return nil, span.Range{}, nil, nil, nil, nil, nil, errors.Errorf("getting file for Identifier: %w", err)
	}
	spn, err := pgf.Mapper.RangeSpan(pRng)
	if err != nil {
		return nil, span.Range{}, nil, nil, nil, nil, nil, err
	}
	rng, err := spn.Range(pgf.Mapper.Converter)
	if err != nil {
		return nil, span.Range{}, nil, nil, nil, nil, nil, err
	}
	src, err := fh.Read()
	if err != nil {
		return nil, span.Range{}, nil, nil, nil, nil, nil, err
	}
	return snapshot.FileSet(), rng, src, pgf.File, pgf.Mapper, pkg.GetTypes(), pkg.GetTypesInfo(), nil
}

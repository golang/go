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
)

type Command struct {
	Name, Title string

	// appliesFn is an optional field to indicate whether or not a command can
	// be applied to the given inputs. If it returns false, we should not
	// suggest this command for these inputs.
	appliesFn AppliesFunc

	// suggestedFixFn is an optional field to generate the edits that the
	// command produces for the given inputs.
	suggestedFixFn SuggestedFixFunc
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
}

var (
	// CommandTest runs `go test` for a specific test function.
	CommandTest = &Command{
		Name: "test",
	}

	// CommandGenerate runs `go generate` for a given directory.
	CommandGenerate = &Command{
		Name: "generate",
	}

	// CommandTidy runs `go mod tidy` for a module.
	CommandTidy = &Command{
		Name: "tidy",
	}

	// CommandVendor runs `go mod vendor` for a module.
	CommandVendor = &Command{
		Name: "vendor",
	}

	// CommandUpgradeDependency upgrades a dependency.
	CommandUpgradeDependency = &Command{
		Name: "upgrade_dependency",
	}

	// CommandRegenerateCgo regenerates cgo definitions.
	CommandRegenerateCgo = &Command{
		Name: "regenerate_cgo",
	}

	// CommandFillStruct is a gopls command to fill a struct with default
	// values.
	CommandFillStruct = &Command{
		Name:           "fill_struct",
		suggestedFixFn: fillstruct.SuggestedFix,
	}

	// CommandUndeclaredName adds a variable declaration for an undeclared
	// name.
	CommandUndeclaredName = &Command{
		Name:           "undeclared_name",
		suggestedFixFn: undeclaredname.SuggestedFix,
	}

	// CommandExtractVariable extracts an expression to a variable.
	CommandExtractVariable = &Command{
		Name:           "extract_variable",
		Title:          "Extract to variable",
		suggestedFixFn: extractVariable,
		appliesFn:      canExtractVariable,
	}

	// CommandExtractFunction extracts statements to a function.
	CommandExtractFunction = &Command{
		Name:           "extract_function",
		Title:          "Extract to function",
		suggestedFixFn: extractFunction,
		appliesFn:      canExtractFunction,
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
func (c *Command) SuggestedFix(ctx context.Context, snapshot Snapshot, fh FileHandle, pRng protocol.Range) ([]protocol.TextDocumentEdit, error) {
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
	pkg, pgh, err := getParsedFile(ctx, snapshot, fh, NarrowestPackageHandle)
	if err != nil {
		return nil, span.Range{}, nil, nil, nil, nil, nil, fmt.Errorf("getting file for Identifier: %w", err)
	}
	file, _, m, _, err := pgh.Cached()
	if err != nil {
		return nil, span.Range{}, nil, nil, nil, nil, nil, err
	}
	spn, err := m.RangeSpan(pRng)
	if err != nil {
		return nil, span.Range{}, nil, nil, nil, nil, nil, err
	}
	rng, err := spn.Range(m.Converter)
	if err != nil {
		return nil, span.Range{}, nil, nil, nil, nil, nil, err
	}
	src, err := fh.Read()
	if err != nil {
		return nil, span.Range{}, nil, nil, nil, nil, nil, err
	}
	fset := snapshot.View().Session().Cache().FileSet()
	return fset, rng, src, file, m, pkg.GetTypes(), pkg.GetTypesInfo(), nil
}

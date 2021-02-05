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

	// Async controls whether the command executes asynchronously.
	Async bool
}

// CommandPrefix is the prefix of all command names gopls uses externally.
const CommandPrefix = "gopls."

// ID adds the CommandPrefix to the command name, in order to avoid
// collisions with other language servers.
func (c Command) ID() string {
	return CommandPrefix + c.Name
}

// SuggestedFixFunc is a function used to get the suggested fixes for a given
// gopls command, some of which are provided by go/analysis.Analyzers. Some of
// the analyzers in internal/lsp/analysis are not efficient enough to include
// suggested fixes with their diagnostics, so we have to compute them
// separately. Such analyzers should provide a function with a signature of
// SuggestedFixFunc.
type SuggestedFixFunc func(fset *token.FileSet, rng span.Range, src []byte, file *ast.File, pkg *types.Package, info *types.Info) (*analysis.SuggestedFix, error)

// Commands are the commands currently supported by gopls.
var Commands = []*Command{
	CommandAddDependency,
	CommandCheckUpgrades,
	CommandExtractFunction,
	CommandExtractVariable,
	CommandFillStruct,
	CommandToggleDetails, // gc_details
	CommandGenerate,
	CommandGenerateGoplsMod,
	CommandGoGetPackage,
	CommandRegenerateCgo,
	CommandRemoveDependency,
	CommandTest,
	CommandTidy,
	CommandUndeclaredName,
	CommandUpdateGoSum,
	CommandUpgradeDependency,
	CommandVendor,
}

var (
	// CommandTest runs `go test` for a specific test function.
	CommandTest = &Command{
		Name:  "test",
		Title: "Run test(s)",
		Async: true,
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

	// CommandGoGetPackage runs `go get` to fetch a package.
	CommandGoGetPackage = &Command{
		Name:  "go_get_package",
		Title: "go get package",
	}

	// CommandUpdateGoSum updates the go.sum file for a module.
	CommandUpdateGoSum = &Command{
		Name:  "update_go_sum",
		Title: "Update go.sum",
	}

	// CommandCheckUpgrades checks for module upgrades.
	CommandCheckUpgrades = &Command{
		Name:  "check_upgrades",
		Title: "Check for upgrades",
	}

	// CommandAddDependency adds a dependency.
	CommandAddDependency = &Command{
		Name:  "add_dependency",
		Title: "Add dependency",
	}

	// CommandUpgradeDependency upgrades a dependency.
	CommandUpgradeDependency = &Command{
		Name:  "upgrade_dependency",
		Title: "Upgrade dependency",
	}

	// CommandRemoveDependency removes a dependency.
	CommandRemoveDependency = &Command{
		Name:  "remove_dependency",
		Title: "Remove dependency",
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
		Name:  "fill_struct",
		Title: "Fill struct",
	}

	// CommandUndeclaredName adds a variable declaration for an undeclared
	// name.
	CommandUndeclaredName = &Command{
		Name:  "undeclared_name",
		Title: "Undeclared name",
	}

	// CommandExtractVariable extracts an expression to a variable.
	CommandExtractVariable = &Command{
		Name:  "extract_variable",
		Title: "Extract to variable",
	}

	// CommandExtractFunction extracts statements to a function.
	CommandExtractFunction = &Command{
		Name:  "extract_function",
		Title: "Extract to function",
	}

	// CommandGenerateGoplsMod (re)generates the gopls.mod file.
	CommandGenerateGoplsMod = &Command{
		Name:  "generate_gopls_mod",
		Title: "Generate gopls.mod",
	}
)

// suggestedFixes maps a suggested fix command id to its handler.
var suggestedFixes = map[string]SuggestedFixFunc{
	CommandFillStruct.ID():      fillstruct.SuggestedFix,
	CommandUndeclaredName.ID():  undeclaredname.SuggestedFix,
	CommandExtractVariable.ID(): extractVariable,
	CommandExtractFunction.ID(): extractFunction,
}

// ApplyFix applies the command's suggested fix to the given file and
// range, returning the resulting edits.
func ApplyFix(ctx context.Context, cmdid string, snapshot Snapshot, fh VersionedFileHandle, pRng protocol.Range) ([]protocol.TextDocumentEdit, error) {
	handler, ok := suggestedFixes[cmdid]
	if !ok {
		return nil, fmt.Errorf("no suggested fix function for %s", cmdid)
	}
	fset, rng, src, file, m, pkg, info, err := getAllSuggestedFixInputs(ctx, snapshot, fh, pRng)
	if err != nil {
		return nil, err
	}
	fix, err := handler(fset, rng, src, file, pkg, info)
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
			TextDocument: protocol.OptionalVersionedTextDocumentIdentifier{
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
	rng, err := pgf.Mapper.RangeToSpanRange(pRng)
	if err != nil {
		return nil, span.Range{}, nil, nil, nil, nil, nil, err
	}
	return snapshot.FileSet(), rng, pgf.Src, pgf.File, pgf.Mapper, pkg.GetTypes(), pkg.GetTypesInfo(), nil
}

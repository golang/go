// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package unusedvariable defines an analyzer that checks for unused variables.
package unusedvariable

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/format"
	"go/token"
	"go/types"
	"strings"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/ast/astutil"
)

const Doc = `check for unused variables

The unusedvariable analyzer suggests fixes for unused variables errors.
`

var Analyzer = &analysis.Analyzer{
	Name:             "unusedvariable",
	Doc:              Doc,
	Requires:         []*analysis.Analyzer{},
	Run:              run,
	RunDespiteErrors: true, // an unusedvariable diagnostic is a compile error
}

// The suffix for this error message changed in Go 1.20.
var unusedVariableSuffixes = []string{" declared and not used", " declared but not used"}

func run(pass *analysis.Pass) (interface{}, error) {
	for _, typeErr := range pass.TypeErrors {
		for _, suffix := range unusedVariableSuffixes {
			if strings.HasSuffix(typeErr.Msg, suffix) {
				varName := strings.TrimSuffix(typeErr.Msg, suffix)
				err := runForError(pass, typeErr, varName)
				if err != nil {
					return nil, err
				}
			}
		}
	}

	return nil, nil
}

func runForError(pass *analysis.Pass, err types.Error, name string) error {
	var file *ast.File
	for _, f := range pass.Files {
		if f.Pos() <= err.Pos && err.Pos < f.End() {
			file = f
			break
		}
	}
	if file == nil {
		return nil
	}

	path, _ := astutil.PathEnclosingInterval(file, err.Pos, err.Pos)
	if len(path) < 2 {
		return nil
	}

	ident, ok := path[0].(*ast.Ident)
	if !ok || ident.Name != name {
		return nil
	}

	diag := analysis.Diagnostic{
		Pos:     ident.Pos(),
		End:     ident.End(),
		Message: err.Msg,
	}

	for i := range path {
		switch stmt := path[i].(type) {
		case *ast.ValueSpec:
			// Find GenDecl to which offending ValueSpec belongs.
			if decl, ok := path[i+1].(*ast.GenDecl); ok {
				fixes := removeVariableFromSpec(pass, path, stmt, decl, ident)
				// fixes may be nil
				if len(fixes) > 0 {
					diag.SuggestedFixes = fixes
					pass.Report(diag)
				}
			}

		case *ast.AssignStmt:
			if stmt.Tok != token.DEFINE {
				continue
			}

			containsIdent := false
			for _, expr := range stmt.Lhs {
				if expr == ident {
					containsIdent = true
				}
			}
			if !containsIdent {
				continue
			}

			fixes := removeVariableFromAssignment(pass, path, stmt, ident)
			// fixes may be nil
			if len(fixes) > 0 {
				diag.SuggestedFixes = fixes
				pass.Report(diag)
			}
		}
	}

	return nil
}

func removeVariableFromSpec(pass *analysis.Pass, path []ast.Node, stmt *ast.ValueSpec, decl *ast.GenDecl, ident *ast.Ident) []analysis.SuggestedFix {
	newDecl := new(ast.GenDecl)
	*newDecl = *decl
	newDecl.Specs = nil

	for _, spec := range decl.Specs {
		if spec != stmt {
			newDecl.Specs = append(newDecl.Specs, spec)
			continue
		}

		newSpec := new(ast.ValueSpec)
		*newSpec = *stmt
		newSpec.Names = nil

		for _, n := range stmt.Names {
			if n != ident {
				newSpec.Names = append(newSpec.Names, n)
			}
		}

		if len(newSpec.Names) > 0 {
			newDecl.Specs = append(newDecl.Specs, newSpec)
		}
	}

	// decl.End() does not include any comments, so if a comment is present we
	// need to account for it when we delete the statement
	end := decl.End()
	if stmt.Comment != nil && stmt.Comment.End() > end {
		end = stmt.Comment.End()
	}

	// There are no other specs left in the declaration, the whole statement can
	// be deleted
	if len(newDecl.Specs) == 0 {
		// Find parent DeclStmt and delete it
		for _, node := range path {
			if declStmt, ok := node.(*ast.DeclStmt); ok {
				return []analysis.SuggestedFix{
					{
						Message:   suggestedFixMessage(ident.Name),
						TextEdits: deleteStmtFromBlock(path, declStmt),
					},
				}
			}
		}
	}

	var b bytes.Buffer
	if err := format.Node(&b, pass.Fset, newDecl); err != nil {
		return nil
	}

	return []analysis.SuggestedFix{
		{
			Message: suggestedFixMessage(ident.Name),
			TextEdits: []analysis.TextEdit{
				{
					Pos: decl.Pos(),
					// Avoid adding a new empty line
					End:     end + 1,
					NewText: b.Bytes(),
				},
			},
		},
	}
}

func removeVariableFromAssignment(pass *analysis.Pass, path []ast.Node, stmt *ast.AssignStmt, ident *ast.Ident) []analysis.SuggestedFix {
	// The only variable in the assignment is unused
	if len(stmt.Lhs) == 1 {
		// If LHS has only one expression to be valid it has to have 1 expression
		// on RHS
		//
		// RHS may have side effects, preserve RHS
		if exprMayHaveSideEffects(stmt.Rhs[0]) {
			// Delete until RHS
			return []analysis.SuggestedFix{
				{
					Message: suggestedFixMessage(ident.Name),
					TextEdits: []analysis.TextEdit{
						{
							Pos: ident.Pos(),
							End: stmt.Rhs[0].Pos(),
						},
					},
				},
			}
		}

		// RHS does not have any side effects, delete the whole statement
		return []analysis.SuggestedFix{
			{
				Message:   suggestedFixMessage(ident.Name),
				TextEdits: deleteStmtFromBlock(path, stmt),
			},
		}
	}

	// Otherwise replace ident with `_`
	return []analysis.SuggestedFix{
		{
			Message: suggestedFixMessage(ident.Name),
			TextEdits: []analysis.TextEdit{
				{
					Pos:     ident.Pos(),
					End:     ident.End(),
					NewText: []byte("_"),
				},
			},
		},
	}
}

func suggestedFixMessage(name string) string {
	return fmt.Sprintf("Remove variable %s", name)
}

func deleteStmtFromBlock(path []ast.Node, stmt ast.Stmt) []analysis.TextEdit {
	// Find innermost enclosing BlockStmt.
	var block *ast.BlockStmt
	for i := range path {
		if blockStmt, ok := path[i].(*ast.BlockStmt); ok {
			block = blockStmt
			break
		}
	}

	nodeIndex := -1
	for i, blockStmt := range block.List {
		if blockStmt == stmt {
			nodeIndex = i
			break
		}
	}

	// The statement we need to delete was not found in BlockStmt
	if nodeIndex == -1 {
		return nil
	}

	// Delete until the end of the block unless there is another statement after
	// the one we are trying to delete
	end := block.Rbrace
	if nodeIndex < len(block.List)-1 {
		end = block.List[nodeIndex+1].Pos()
	}

	return []analysis.TextEdit{
		{
			Pos: stmt.Pos(),
			End: end,
		},
	}
}

// exprMayHaveSideEffects reports whether the expression may have side effects
// (because it contains a function call or channel receive). We disregard
// runtime panics as well written programs should not encounter them.
func exprMayHaveSideEffects(expr ast.Expr) bool {
	var mayHaveSideEffects bool
	ast.Inspect(expr, func(n ast.Node) bool {
		switch n := n.(type) {
		case *ast.CallExpr: // possible function call
			mayHaveSideEffects = true
			return false
		case *ast.UnaryExpr:
			if n.Op == token.ARROW { // channel receive
				mayHaveSideEffects = true
				return false
			}
		case *ast.FuncLit:
			return false // evaluating what's inside a FuncLit has no effect
		}
		return true
	})

	return mayHaveSideEffects
}

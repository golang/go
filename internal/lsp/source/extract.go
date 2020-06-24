// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"bytes"
	"context"
	"fmt"
	"go/ast"
	"go/format"
	"go/token"
	"go/types"

	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/internal/analysisinternal"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/span"
)

func ExtractVariable(ctx context.Context, snapshot Snapshot, fh FileHandle, protoRng protocol.Range) ([]protocol.TextEdit, error) {
	pkg, pgh, err := getParsedFile(ctx, snapshot, fh, NarrowestPackageHandle)
	if err != nil {
		return nil, fmt.Errorf("ExtractVariable: %v", err)
	}
	file, _, m, _, err := pgh.Cached()
	if err != nil {
		return nil, err
	}
	spn, err := m.RangeSpan(protoRng)
	if err != nil {
		return nil, err
	}
	rng, err := spn.Range(m.Converter)
	if err != nil {
		return nil, err
	}
	path, _ := astutil.PathEnclosingInterval(file, rng.Start, rng.End)
	if len(path) == 0 {
		return nil, nil
	}
	fset := snapshot.View().Session().Cache().FileSet()
	node := path[0]
	tok := fset.File(node.Pos())
	if tok == nil {
		return nil, fmt.Errorf("ExtractVariable: no token.File for %s", fh.URI())
	}
	var content []byte
	if content, err = fh.Read(); err != nil {
		return nil, err
	}
	if rng.Start != node.Pos() || rng.End != node.End() {
		return nil, nil
	}

	// Adjust new variable name until no collisons in scope.
	scopes := collectScopes(pkg, path, node.Pos())
	name := "x0"
	idx := 0
	for !isValidName(name, scopes) {
		idx++
		name = fmt.Sprintf("x%d", idx)
	}

	var assignment string
	expr, ok := node.(ast.Expr)
	if !ok {
		return nil, nil
	}
	// Create new AST node for extracted code
	switch expr.(type) {
	case *ast.BasicLit, *ast.CompositeLit, *ast.IndexExpr,
		*ast.SliceExpr, *ast.UnaryExpr, *ast.BinaryExpr, *ast.SelectorExpr: // TODO: stricter rules for selectorExpr
		assignStmt := &ast.AssignStmt{
			Lhs: []ast.Expr{ast.NewIdent(name)},
			Tok: token.DEFINE,
			Rhs: []ast.Expr{expr},
		}
		var buf bytes.Buffer
		if err = format.Node(&buf, fset, assignStmt); err != nil {
			return nil, err
		}
		assignment = buf.String()
	case *ast.CallExpr: // TODO: find number of return values and do according actions.
		return nil, nil
	default:
		return nil, nil
	}

	insertBeforeStmt := analysisinternal.StmtToInsertVarBefore(path)
	if insertBeforeStmt == nil {
		return nil, nil
	}

	// Convert token.Pos to protcol.Position
	rng = span.NewRange(fset, insertBeforeStmt.Pos(), insertBeforeStmt.End())
	spn, err = rng.Span()
	if err != nil {
		return nil, nil
	}
	beforeStmtStart, err := m.Position(spn.Start())
	if err != nil {
		return nil, nil
	}
	stmtBeforeRng := protocol.Range{
		Start: beforeStmtStart,
		End:   beforeStmtStart,
	}

	// Calculate indentation for insertion
	line := tok.Line(insertBeforeStmt.Pos())
	lineOffset := tok.Offset(tok.LineStart(line))
	stmtOffset := tok.Offset(insertBeforeStmt.Pos())
	indent := content[lineOffset:stmtOffset] // space between these is indentation.

	return []protocol.TextEdit{
		{
			Range:   stmtBeforeRng,
			NewText: assignment + "\n" + string(indent),
		},
		{
			Range:   protoRng,
			NewText: name,
		},
	}, nil
}

// Check for variable collision in scope.
func isValidName(name string, scopes []*types.Scope) bool {
	for _, scope := range scopes {
		if scope == nil {
			continue
		}
		if scope.Lookup(name) != nil {
			return false
		}
	}
	return true
}

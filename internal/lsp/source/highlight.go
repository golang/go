// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"
	"go/ast"

	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/telemetry/log"
	"golang.org/x/tools/internal/telemetry/trace"
	errors "golang.org/x/xerrors"
)

func Highlight(ctx context.Context, snapshot Snapshot, f File, pos protocol.Position) ([]protocol.Range, error) {
	ctx, done := trace.StartSpan(ctx, "source.Highlight")
	defer done()

	cphs, err := snapshot.PackageHandles(ctx, f)
	if err != nil {
		return nil, err
	}
	cph, err := WidestCheckPackageHandle(cphs)
	if err != nil {
		return nil, err
	}
	pkg, err := cph.Check(ctx)
	if err != nil {
		return nil, err
	}
	var ph ParseGoHandle
	for _, file := range pkg.CompiledGoFiles() {
		if file.File().Identity().URI == f.URI() {
			ph = file
		}
	}
	if ph == nil {
		return nil, errors.Errorf("no ParseGoHandle for %s", f.URI())
	}
	file, m, _, err := ph.Parse(ctx)
	if err != nil {
		return nil, err
	}
	spn, err := m.PointSpan(pos)
	if err != nil {
		return nil, err
	}
	rng, err := spn.Range(m.Converter)
	if err != nil {
		return nil, err
	}
	path, _ := astutil.PathEnclosingInterval(file, rng.Start, rng.Start)
	if len(path) == 0 {
		return nil, errors.Errorf("no enclosing position found for %v:%v", int(pos.Line), int(pos.Character))
	}

	switch path[0].(type) {
	case *ast.Ident:
		return highlightIdentifiers(ctx, snapshot, m, path, pkg)
	case *ast.BranchStmt, *ast.ForStmt, *ast.RangeStmt:
		return highlightControlFlow(ctx, snapshot, m, path)
	}

	// If the cursor is in an unidentified area, return empty results.
	return nil, nil
}

func highlightControlFlow(ctx context.Context, snapshot Snapshot, m *protocol.ColumnMapper, path []ast.Node) ([]protocol.Range, error) {
	// Reverse walk the path till we get to the for loop.
	var loop ast.Node
Outer:
	for _, n := range path {
		switch n.(type) {
		case *ast.ForStmt, *ast.RangeStmt:
			loop = n
			break Outer
		}
	}
	if loop == nil {
		// Cursor is not in a for loop.
		return nil, nil
	}

	var result []protocol.Range

	// Add the for statement.
	forStmt, err := posToRange(snapshot.View(), m, loop.Pos(), loop.Pos()+3)
	if err != nil {
		return nil, err
	}
	rng, err := forStmt.Range()
	if err != nil {
		return nil, err
	}
	result = append(result, rng)

	ast.Inspect(loop, func(n ast.Node) bool {
		// Don't traverse any other for loops.
		switch n.(type) {
		case *ast.ForStmt, *ast.RangeStmt:
			return loop == n
		}

		if n, ok := n.(*ast.BranchStmt); ok {
			// Add all branch statements in same scope as the identified one.
			rng, err := nodeToProtocolRange(ctx, snapshot.View(), m, n)
			if err != nil {
				log.Error(ctx, "Error getting range for node", err)
				return false
			}
			result = append(result, rng)
		}
		return true
	})
	return result, nil
}

func highlightIdentifiers(ctx context.Context, snapshot Snapshot, m *protocol.ColumnMapper, path []ast.Node, pkg Package) ([]protocol.Range, error) {
	var result []protocol.Range
	id, ok := path[0].(*ast.Ident)
	if !ok {
		return nil, errors.Errorf("highlightIdentifiers called with an ast.Node of type %T", id)
	}

	idObj := pkg.GetTypesInfo().ObjectOf(id)
	ast.Inspect(path[len(path)-1], func(node ast.Node) bool {
		n, ok := node.(*ast.Ident)
		if !ok {
			return true
		}
		if n.Name != id.Name || n.Obj != id.Obj {
			return false
		}
		if nObj := pkg.GetTypesInfo().ObjectOf(n); nObj != idObj {
			return false
		}

		if rng, err := nodeToProtocolRange(ctx, snapshot.View(), m, n); err == nil {
			result = append(result, rng)
		} else {
			log.Error(ctx, "Error getting range for node", err)
		}
		return false
	})
	return result, nil
}

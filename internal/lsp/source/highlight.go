// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"
	"go/ast"

	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/telemetry/trace"
	errors "golang.org/x/xerrors"
)

func Highlight(ctx context.Context, view View, uri span.URI, pos protocol.Position) ([]protocol.Range, error) {
	ctx, done := trace.StartSpan(ctx, "source.Highlight")
	defer done()

	f, err := view.GetFile(ctx, uri)
	if err != nil {
		return nil, err
	}
	_, cphs, err := view.CheckPackageHandles(ctx, f)
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
	result := []protocol.Range{}
	id, ok := path[0].(*ast.Ident)
	if !ok {
		// If the cursor is not within an identifier, return empty results.
		return result, nil
	}
	idObj := pkg.GetTypesInfo().ObjectOf(id)

	ast.Inspect(path[len(path)-1], func(node ast.Node) bool {
		n, ok := node.(*ast.Ident)
		if !ok {
			return true
		}
		if n.Name != id.Name {
			return true
		}
		if n.Obj != id.Obj {
			return true
		}

		nodeObj := pkg.GetTypesInfo().ObjectOf(n)
		if nodeObj != idObj {
			return false
		}

		if rng, err := nodeToProtocolRange(ctx, view, m, n); err == nil {
			result = append(result, rng)
		}
		return true
	})
	return result, nil
}

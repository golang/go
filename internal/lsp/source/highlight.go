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
	ph := view.Session().Cache().ParseGoHandle(f.Handle(ctx), ParseFull)
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
		return nil, errors.Errorf("no enclosing position found for %f:%f", pos.Line, pos.Character)
	}
	id, ok := path[0].(*ast.Ident)
	if !ok {
		return nil, errors.Errorf("%f:%f is not an identifier", pos.Line, pos.Character)
	}
	var result []protocol.Range
	if id.Obj != nil {
		ast.Inspect(path[len(path)-1], func(n ast.Node) bool {
			if n, ok := n.(*ast.Ident); ok && n.Obj == id.Obj {
				rng, err := nodeToProtocolRange(ctx, view, m, n)
				if err == nil {
					result = append(result, rng)
				}
			}
			return true
		})
	}
	return result, nil
}

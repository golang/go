// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"
	"fmt"
	"go/ast"
	"go/token"

	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/internal/lsp/telemetry/trace"
	"golang.org/x/tools/internal/span"
)

func Highlight(ctx context.Context, f GoFile, pos token.Pos) ([]span.Span, error) {
	ctx, done := trace.StartSpan(ctx, "source.Highlight")
	defer done()

	file, err := f.GetAST(ctx, ParseFull)
	if file == nil {
		return nil, err
	}
	fset := f.FileSet()
	path, _ := astutil.PathEnclosingInterval(file, pos, pos)
	if len(path) == 0 {
		return nil, fmt.Errorf("no enclosing position found for %s", fset.Position(pos))
	}
	id, ok := path[0].(*ast.Ident)
	if !ok {
		return nil, fmt.Errorf("%s is not an identifier", fset.Position(pos))
	}
	var result []span.Span
	if id.Obj != nil {
		ast.Inspect(path[len(path)-1], func(n ast.Node) bool {
			if n, ok := n.(*ast.Ident); ok && n.Obj == id.Obj {
				s, err := nodeSpan(n, fset)
				if err == nil {
					result = append(result, s)
				}
			}
			return true
		})
	}
	return result, nil
}

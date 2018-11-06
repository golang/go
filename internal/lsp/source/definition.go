// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"
	"fmt"
	"go/ast"
	"go/token"

	"golang.org/x/tools/go/ast/astutil"
)

func Definition(ctx context.Context, f *File, pos token.Pos) (Range, error) {
	fAST, err := f.GetAST()
	if err != nil {
		return Range{}, err
	}
	pkg, err := f.GetPackage()
	if err != nil {
		return Range{}, err
	}
	ident, err := findIdentifier(fAST, pos)
	if err != nil {
		return Range{}, err
	}
	if ident == nil {
		return Range{}, fmt.Errorf("definition was not a valid identifier")
	}
	obj := pkg.TypesInfo.ObjectOf(ident)
	if obj == nil {
		return Range{}, fmt.Errorf("no object")
	}
	return Range{
		Start: obj.Pos(),
		End:   obj.Pos() + token.Pos(len([]byte(obj.Name()))), // TODO: use real range of obj
	}, nil
}

// findIdentifier returns the ast.Ident for a position
// in a file, accounting for a potentially incomplete selector.
func findIdentifier(f *ast.File, pos token.Pos) (*ast.Ident, error) {
	path, _ := astutil.PathEnclosingInterval(f, pos, pos)
	if path == nil {
		return nil, fmt.Errorf("can't find node enclosing position")
	}
	// If the position is not an identifier but immediately follows
	// an identifier or selector period (as is common when
	// requesting a completion), use the path to the preceding node.
	if ident, ok := path[0].(*ast.Ident); ok {
		return ident, nil
	}
	path, _ = astutil.PathEnclosingInterval(f, pos-1, pos-1)
	if path == nil {
		return nil, nil
	}
	switch prev := path[0].(type) {
	case *ast.Ident:
		return prev, nil
	case *ast.SelectorExpr:
		return prev.Sel, nil
	}
	return nil, nil
}

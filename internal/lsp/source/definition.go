// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"
	"fmt"
	"go/ast"
	"go/token"
	"go/types"

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
	i, err := findIdentifier(fAST, pos)
	if err != nil {
		return Range{}, err
	}
	if i.ident == nil {
		return Range{}, fmt.Errorf("definition was not a valid identifier")
	}
	obj := pkg.TypesInfo.ObjectOf(i.ident)
	if obj == nil {
		return Range{}, fmt.Errorf("no object")
	}
	if i.wasEmbeddedField {
		// the original position was on the embedded field declaration
		// so we try to dig out the type and jump to that instead
		if v, ok := obj.(*types.Var); ok {
			if n, ok := v.Type().(*types.Named); ok {
				obj = n.Obj()
			}
		}
	}
	return Range{
		Start: obj.Pos(),
		End:   obj.Pos() + token.Pos(len([]byte(obj.Name()))), // TODO: use real range of obj
	}, nil
}

// ident returns the ident plus any extra information needed
type ident struct {
	ident            *ast.Ident
	wasEmbeddedField bool
}

// findIdentifier returns the ast.Ident for a position
// in a file, accounting for a potentially incomplete selector.
func findIdentifier(f *ast.File, pos token.Pos) (ident, error) {
	m, err := checkIdentifier(f, pos)
	if err != nil {
		return ident{}, err
	}
	if m.ident != nil {
		return m, nil
	}
	// If the position is not an identifier but immediately follows
	// an identifier or selector period (as is common when
	// requesting a completion), use the path to the preceding node.
	return checkIdentifier(f, pos-1)
}

// checkIdentifier checks a single position for a potential identifier.
func checkIdentifier(f *ast.File, pos token.Pos) (ident, error) {
	path, _ := astutil.PathEnclosingInterval(f, pos, pos)
	result := ident{}
	if path == nil {
		return result, fmt.Errorf("can't find node enclosing position")
	}
	switch node := path[0].(type) {
	case *ast.Ident:
		result.ident = node
	case *ast.SelectorExpr:
		result.ident = node.Sel
	}
	if result.ident != nil {
		for _, n := range path[1:] {
			if field, ok := n.(*ast.Field); ok {
				result.wasEmbeddedField = len(field.Names) == 0
			}
		}
	}
	return result, nil
}

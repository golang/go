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
	"golang.org/x/tools/internal/span"
)

// IdentifierInfo holds information about an identifier in Go source.
type IdentifierInfo struct {
	Name  string
	Range span.Range
	File  File
	Type  struct {
		Range  span.Range
		Object types.Object
	}
	Declaration struct {
		Range  span.Range
		Object types.Object
	}

	ident            *ast.Ident
	wasEmbeddedField bool
}

// Identifier returns identifier information for a position
// in a file, accounting for a potentially incomplete selector.
func Identifier(ctx context.Context, v View, f File, pos token.Pos) (*IdentifierInfo, error) {
	if result, err := identifier(ctx, v, f, pos); err != nil || result != nil {
		return result, err
	}
	// If the position is not an identifier but immediately follows
	// an identifier or selector period (as is common when
	// requesting a completion), use the path to the preceding node.
	result, err := identifier(ctx, v, f, pos-1)
	if result == nil && err == nil {
		err = fmt.Errorf("no identifier found")
	}
	return result, err
}

func (i *IdentifierInfo) Hover(ctx context.Context, q types.Qualifier) (string, error) {
	if q == nil {
		fAST := i.File.GetAST(ctx)
		pkg := i.File.GetPackage(ctx)
		q = qualifier(fAST, pkg.GetTypes(), pkg.GetTypesInfo())
	}
	return types.ObjectString(i.Declaration.Object, q), nil
}

// identifier checks a single position for a potential identifier.
func identifier(ctx context.Context, v View, f File, pos token.Pos) (*IdentifierInfo, error) {
	fAST := f.GetAST(ctx)
	pkg := f.GetPackage(ctx)
	if pkg.IsIllTyped() {
		return nil, fmt.Errorf("package for %s is ill typed", f.URI())
	}
	path, _ := astutil.PathEnclosingInterval(fAST, pos, pos)
	result := &IdentifierInfo{
		File: f,
	}
	if path == nil {
		return nil, fmt.Errorf("can't find node enclosing position")
	}
	switch node := path[0].(type) {
	case *ast.Ident:
		result.ident = node
	case *ast.SelectorExpr:
		result.ident = node.Sel
	}
	if result.ident == nil {
		return nil, nil
	}
	for _, n := range path[1:] {
		if field, ok := n.(*ast.Field); ok {
			result.wasEmbeddedField = len(field.Names) == 0
		}
	}
	result.Name = result.ident.Name
	result.Range = span.NewRange(v.FileSet(), result.ident.Pos(), result.ident.End())
	result.Declaration.Object = pkg.GetTypesInfo().ObjectOf(result.ident)
	if result.Declaration.Object == nil {
		return nil, fmt.Errorf("no object for ident %v", result.Name)
	}
	if result.wasEmbeddedField {
		// The original position was on the embedded field declaration, so we
		// try to dig out the type and jump to that instead.
		if v, ok := result.Declaration.Object.(*types.Var); ok {
			if n, ok := v.Type().(*types.Named); ok {
				result.Declaration.Object = n.Obj()
			}
		}
	}
	var err error
	if result.Declaration.Range, err = objToRange(ctx, v, result.Declaration.Object); err != nil {
		return nil, err
	}
	typ := pkg.GetTypesInfo().TypeOf(result.ident)
	if typ == nil {
		return nil, fmt.Errorf("no type for %s", result.Name)
	}
	result.Type.Object = typeToObject(typ)
	if result.Type.Object != nil {
		// Identifiers with the type "error" are a special case with no position.
		if types.IsInterface(result.Type.Object.Type()) && result.Type.Object.Pkg() == nil && result.Type.Object.Name() == "error" {
			return result, nil
		}
		if result.Type.Range, err = objToRange(ctx, v, result.Type.Object); err != nil {
			return nil, err
		}
	}
	return result, nil
}

func typeToObject(typ types.Type) types.Object {
	switch typ := typ.(type) {
	case *types.Named:
		return typ.Obj()
	case *types.Pointer:
		return typeToObject(typ.Elem())
	default:
		return nil
	}
}

func objToRange(ctx context.Context, v View, obj types.Object) (span.Range, error) {
	p := obj.Pos()
	if !p.IsValid() {
		return span.Range{}, fmt.Errorf("invalid position for %v", obj.Name())
	}
	return span.NewRange(v.FileSet(), p, p+token.Pos(len(obj.Name()))), nil
}

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

// IdentifierInfo holds information about an identifier in Go source.
type IdentifierInfo struct {
	Name  string
	Range Range
	File  File
	Type  struct {
		Range  Range
		Object types.Object
	}
	Declaration struct {
		Range  Range
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
		q = qualifier(fAST, pkg.Types, pkg.TypesInfo)
	}
	return types.ObjectString(i.Declaration.Object, q), nil
}

// identifier checks a single position for a potential identifier.
func identifier(ctx context.Context, v View, f File, pos token.Pos) (*IdentifierInfo, error) {
	fAST := f.GetAST(ctx)
	pkg := f.GetPackage(ctx)
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
	result.Range = Range{Start: result.ident.Pos(), End: result.ident.End()}
	result.Declaration.Object = pkg.TypesInfo.ObjectOf(result.ident)
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
	typ := pkg.TypesInfo.TypeOf(result.ident)
	if typ == nil {
		return nil, fmt.Errorf("no type for %s", result.Name)
	}
	result.Type.Object = typeToObject(typ)
	if result.Type.Object != nil {
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

func objToRange(ctx context.Context, v View, obj types.Object) (Range, error) {
	p := obj.Pos()
	if !p.IsValid() {
		return Range{}, fmt.Errorf("invalid position for %v", obj.Name())
	}
	return Range{
		Start: p,
		End:   p + token.Pos(identifierLen(obj.Name())),
	}, nil
}

// TODO: This needs to be fixed to address golang.org/issue/29149.
func identifierLen(ident string) int {
	return len([]byte(ident))
}

// this functionality was borrowed from the analysisutil package
func lineStart(f *token.File, line int) token.Pos {
	// Use binary search to find the start offset of this line.
	//
	// TODO(rstambler): eventually replace this function with the
	// simpler and more efficient (*go/token.File).LineStart, added
	// in go1.12.

	min := 0        // inclusive
	max := f.Size() // exclusive
	for {
		offset := (min + max) / 2
		pos := f.Pos(offset)
		posn := f.Position(pos)
		if posn.Line == line {
			return pos - (token.Pos(posn.Column) - 1)
		}

		if min+1 >= max {
			return token.NoPos
		}

		if posn.Line < line {
			min = offset
		} else {
			max = offset
		}
	}
}

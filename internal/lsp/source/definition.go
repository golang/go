// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"bytes"
	"context"
	"fmt"
	"go/ast"
	"go/token"
	"go/types"

	"golang.org/x/tools/go/ast/astutil"
)

func Definition(ctx context.Context, v View, f File, pos token.Pos) (Range, error) {
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
		return Range{}, fmt.Errorf("not a valid identifier")
	}
	obj := pkg.TypesInfo.ObjectOf(i.ident)
	if obj == nil {
		return Range{}, fmt.Errorf("no object")
	}
	if i.wasEmbeddedField {
		// The original position was on the embedded field declaration, so we
		// try to dig out the type and jump to that instead.
		if v, ok := obj.(*types.Var); ok {
			if n, ok := v.Type().(*types.Named); ok {
				obj = n.Obj()
			}
		}
	}
	return objToRange(ctx, v, obj)
}

func TypeDefinition(ctx context.Context, v View, f File, pos token.Pos) (Range, error) {
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
		return Range{}, fmt.Errorf("not a valid identifier")
	}
	typ := pkg.TypesInfo.TypeOf(i.ident)
	if typ == nil {
		return Range{}, fmt.Errorf("no type for %s", i.ident.Name)
	}
	obj := typeToObject(typ)
	if obj == nil {
		return Range{}, fmt.Errorf("no object for type %s", typ.String())
	}
	return objToRange(ctx, v, obj)
}

func typeToObject(typ types.Type) (obj types.Object) {
	switch typ := typ.(type) {
	case *types.Named:
		obj = typ.Obj()
	case *types.Pointer:
		obj = typeToObject(typ.Elem())
	}
	return obj
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

func objToRange(ctx context.Context, v View, obj types.Object) (Range, error) {
	p := obj.Pos()
	if !p.IsValid() {
		return Range{}, fmt.Errorf("invalid position for %v", obj.Name())
	}
	tok := v.FileSet().File(p)
	pos := tok.Position(p)
	if pos.Column == 1 {
		// We do not have full position information because exportdata does not
		// store the column. For now, we attempt to read the original source
		// and find the identifier within the line. If we find it, we patch the
		// column to match its offset.
		//
		// TODO: If we parse from source, we will never need this hack.
		f, err := v.GetFile(ctx, ToURI(pos.Filename))
		if err != nil {
			goto Return
		}
		src, err := f.Read()
		if err != nil {
			goto Return
		}
		tok, err := f.GetToken()
		if err != nil {
			goto Return
		}
		start := lineStart(tok, pos.Line)
		offset := tok.Offset(start)
		col := bytes.Index(src[offset:], []byte(obj.Name()))
		p = tok.Pos(offset + col)
	}
Return:
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
	// TODO(adonovan): eventually replace this function with the
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

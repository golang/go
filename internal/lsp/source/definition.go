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
	"io/ioutil"

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
	return objToRange(f.view.Config.Fset, obj), nil
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

func objToRange(fSet *token.FileSet, obj types.Object) Range {
	p := obj.Pos()
	f := fSet.File(p)
	pos := f.Position(p)
	if pos.Column == 1 {
		// Column is 1, so we probably do not have full position information
		// Currently exportdata does not store the column.
		// For now we attempt to read the original source and  find the identifier
		// within the line. If we find it we patch the column to match its offset.
		// TODO: we have probably already added the full data for the file to the
		// fileset, we ought to track it rather than adding it over and over again
		// TODO: if we parse from source, we will never need this hack
		if src, err := ioutil.ReadFile(pos.Filename); err == nil {
			newF := fSet.AddFile(pos.Filename, -1, len(src))
			newF.SetLinesForContent(src)
			lineStart := lineStart(newF, pos.Line)
			offset := newF.Offset(lineStart)
			col := bytes.Index(src[offset:], []byte(obj.Name()))
			p = newF.Pos(offset + col)
		}
	}
	return Range{
		Start: p,
		End:   p + token.Pos(len([]byte(obj.Name()))), // TODO: use real range of obj
	}
}

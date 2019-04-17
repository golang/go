// Copyright 201p The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"bytes"
	"context"
	"fmt"
	"go/ast"
	"go/format"
	"go/token"
	"go/types"
)

func (i *IdentifierInfo) Hover(ctx context.Context, q types.Qualifier, enhancedHover bool) (string, string, error) {
	file := i.File.GetAST(ctx)
	if q == nil {
		pkg := i.File.GetPackage(ctx)
		q = qualifier(file, pkg.GetTypes(), pkg.GetTypesInfo())
	}
	// TODO(rstambler): Remove this configuration when hover behavior is stable.
	if enhancedHover {
		switch obj := i.Declaration.Object.(type) {
		case *types.TypeName:
			if node, ok := i.Declaration.Node.(*ast.GenDecl); ok {
				if decl, doc, err := formatTypeName(i.File.GetFileSet(ctx), node, obj, q); err == nil {
					return decl, doc, nil
				} else {
					// Swallow errors so we can return a best-effort response using types.TypeString.
					i.File.View().Logger().Errorf(ctx, "no hover for TypeName %v: %v", obj.Name(), err)
				}
			}
			return types.TypeString(obj.Type(), q), "", nil
		default:
			return types.ObjectString(obj, q), "", nil
		}
	}
	return types.ObjectString(i.Declaration.Object, q), "", nil
}

func formatTypeName(fset *token.FileSet, decl *ast.GenDecl, obj *types.TypeName, q types.Qualifier) (string, string, error) {
	if types.IsInterface(obj.Type()) {
		return "", "", fmt.Errorf("no support for interfaces yet")
	}
	switch t := obj.Type().(type) {
	case *types.Struct:
		return formatStructType(fset, decl, t)
	case *types.Named:
		if under, ok := t.Underlying().(*types.Struct); ok {
			return formatStructType(fset, decl, under)
		}
	}
	return "", "", fmt.Errorf("no supported for %v, which is of type %T", obj.Name(), obj.Type())
}

func formatStructType(fset *token.FileSet, decl *ast.GenDecl, typ *types.Struct) (string, string, error) {
	if len(decl.Specs) != 1 {
		return "", "", fmt.Errorf("expected 1 TypeSpec got %v", len(decl.Specs))
	}
	b := bytes.NewBuffer(nil)
	if err := format.Node(b, fset, decl.Specs[0]); err != nil {
		return "", "", err
	}
	doc := decl.Doc.Text()
	return b.String(), doc, nil

}

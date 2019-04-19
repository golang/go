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

// formatter returns the a hover value formatted with its documentation.
type formatter func(interface{}, *ast.CommentGroup) (string, error)

func (i *IdentifierInfo) Hover(ctx context.Context, qf types.Qualifier, enhancedHover, markdownSupported bool) (string, error) {
	file := i.File.GetAST(ctx)
	if qf == nil {
		pkg := i.File.GetPackage(ctx)
		qf = qualifier(file, pkg.GetTypes(), pkg.GetTypesInfo())
	}
	b := bytes.NewBuffer(nil)
	f := func(x interface{}, c *ast.CommentGroup) (string, error) {
		return writeHover(x, i.File.GetFileSet(ctx), b, c, markdownSupported, qf)
	}
	obj := i.Declaration.Object
	// TODO(rstambler): Remove this configuration when hover behavior is stable.
	if enhancedHover {
		switch node := i.Declaration.Node.(type) {
		case *ast.GenDecl:
			switch obj := obj.(type) {
			case *types.TypeName, *types.Var, *types.Const, *types.Func:
				return formatGenDecl(node, obj, obj.Type(), f)
			}
		case *ast.FuncDecl:
			if _, ok := obj.(*types.Func); ok {
				return f(obj, node.Doc)
			}
		}
	}
	return f(obj, nil)
}

func formatGenDecl(node *ast.GenDecl, obj types.Object, typ types.Type, f formatter) (string, error) {
	if _, ok := typ.(*types.Named); ok {
		switch typ.Underlying().(type) {
		case *types.Interface, *types.Struct:
			return formatGenDecl(node, obj, typ.Underlying(), f)
		}
	}
	var spec ast.Spec
	for _, s := range node.Specs {
		if s.Pos() <= obj.Pos() && obj.Pos() <= s.End() {
			spec = s
			break
		}
	}
	if spec == nil {
		return "", fmt.Errorf("no spec for node %v at position %v", node, obj.Pos())
	}
	// If we have a field or method.
	switch obj.(type) {
	case *types.Var, *types.Const, *types.Func:
		return formatVar(spec, obj, f)
	}
	// Handle types.
	switch spec := spec.(type) {
	case *ast.TypeSpec:
		// If multiple types are declared in the same block.
		if len(node.Specs) > 1 {
			return f(spec.Type, spec.Doc)
		} else {
			return f(spec, node.Doc)
		}
	case *ast.ValueSpec:
		return f(spec, spec.Doc)
	case *ast.ImportSpec:
		return f(spec, spec.Doc)
	}
	return "", fmt.Errorf("unable to format spec %v (%T)", spec, spec)
}

func formatVar(node ast.Spec, obj types.Object, f formatter) (string, error) {
	var fieldList *ast.FieldList
	if spec, ok := node.(*ast.TypeSpec); ok {
		switch t := spec.Type.(type) {
		case *ast.StructType:
			fieldList = t.Fields
		case *ast.InterfaceType:
			fieldList = t.Methods
		}
	}
	// If we have a struct or interface declaration,
	// we need to match the object to the corresponding field or method.
	if fieldList != nil {
		for i := 0; i < fieldList.NumFields(); i++ {
			field := fieldList.List[i]
			if field.Pos() <= obj.Pos() && obj.Pos() <= field.End() {
				if field.Doc.Text() != "" {
					return f(obj, field.Doc)
				} else if field.Comment.Text() != "" {
					return f(obj, field.Comment)
				}
			}
		}
	}
	// If we weren't able to find documentation for the object.
	return f(obj, nil)
}

// writeHover writes the hover for a given node and its documentation.
func writeHover(x interface{}, fset *token.FileSet, b *bytes.Buffer, c *ast.CommentGroup, markdownSupported bool, qf types.Qualifier) (string, error) {
	if c != nil {
		b.WriteString(c.Text())
		b.WriteRune('\n')
	}
	if markdownSupported {
		b.WriteString("```go\n")
	}
	switch x := x.(type) {
	case ast.Node:
		if err := format.Node(b, fset, x); err != nil {
			return "", err
		}
	case types.Object:
		b.WriteString(types.ObjectString(x, qf))
	}
	if markdownSupported {
		b.WriteString("\n```")
	}
	return b.String(), nil
}

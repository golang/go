// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"
	"fmt"
	"go/ast"
	"go/doc"
	"go/format"
	"go/token"
	"go/types"
	"strings"
)

type documentation struct {
	source  interface{}
	comment *ast.CommentGroup
}

func (i *IdentifierInfo) Hover(ctx context.Context, markdownSupported, wantComments bool) (string, error) {
	h, err := i.decl.hover(ctx)
	if err != nil {
		return "", err
	}
	c := h.comment
	if !wantComments {
		c = nil
	}
	var b strings.Builder
	return writeHover(h.source, i.File.FileSet(), &b, c, markdownSupported, i.qf)
}

func (d declaration) hover(ctx context.Context) (*documentation, error) {
	obj := d.obj
	switch node := d.node.(type) {
	case *ast.GenDecl:
		switch obj := obj.(type) {
		case *types.TypeName, *types.Var, *types.Const, *types.Func:
			return formatGenDecl(node, obj, obj.Type())
		}
	case *ast.TypeSpec:
		if obj.Parent() == types.Universe {
			if obj.Name() == "error" {
				return &documentation{node, nil}, nil
			}
			return &documentation{node.Name, nil}, nil // comments not needed for builtins
		}
	case *ast.FuncDecl:
		switch obj.(type) {
		case *types.Func:
			return &documentation{obj, node.Doc}, nil
		case *types.Builtin:
			return &documentation{node.Type, node.Doc}, nil
		}
	}
	return &documentation{obj, nil}, nil
}

func formatGenDecl(node *ast.GenDecl, obj types.Object, typ types.Type) (*documentation, error) {
	if _, ok := typ.(*types.Named); ok {
		switch typ.Underlying().(type) {
		case *types.Interface, *types.Struct:
			return formatGenDecl(node, obj, typ.Underlying())
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
		return nil, fmt.Errorf("no spec for node %v at position %v", node, obj.Pos())
	}
	// If we have a field or method.
	switch obj.(type) {
	case *types.Var, *types.Const, *types.Func:
		return formatVar(spec, obj)
	}
	// Handle types.
	switch spec := spec.(type) {
	case *ast.TypeSpec:
		if len(node.Specs) > 1 {
			// If multiple types are declared in the same block.
			return &documentation{spec.Type, spec.Doc}, nil
		} else {
			return &documentation{spec, node.Doc}, nil
		}
	case *ast.ValueSpec:
		return &documentation{spec, spec.Doc}, nil
	case *ast.ImportSpec:
		return &documentation{spec, spec.Doc}, nil
	}
	return nil, fmt.Errorf("unable to format spec %v (%T)", spec, spec)
}

func formatVar(node ast.Spec, obj types.Object) (*documentation, error) {
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
		for i := 0; i < len(fieldList.List); i++ {
			field := fieldList.List[i]
			if field.Pos() <= obj.Pos() && obj.Pos() <= field.End() {
				if field.Doc.Text() != "" {
					return &documentation{obj, field.Doc}, nil
				} else if field.Comment.Text() != "" {
					return &documentation{obj, field.Comment}, nil
				}
			}
		}
	}
	// If we weren't able to find documentation for the object.
	return &documentation{obj, nil}, nil
}

// writeHover writes the hover for a given node and its documentation.
func writeHover(x interface{}, fset *token.FileSet, b *strings.Builder, c *ast.CommentGroup, markdownSupported bool, qf types.Qualifier) (string, error) {
	if c != nil {
		// TODO(rstambler): Improve conversion from Go docs to markdown.
		b.WriteString(formatDocumentation(c))
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

func formatDocumentation(c *ast.CommentGroup) string {
	if c == nil {
		return ""
	}
	return doc.Synopsis(c.Text())
}

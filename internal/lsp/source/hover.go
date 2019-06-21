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
	"go/types"
	"strings"

	"golang.org/x/tools/internal/lsp/telemetry/trace"
)

type documentation struct {
	source  interface{}
	comment *ast.CommentGroup
}

type HoverKind int

const (
	NoDocumentation = HoverKind(iota)
	SynopsisDocumentation
	FullDocumentation

	// TODO: Support a single-line hover mode for clients like Vim.
	singleLine
)

func (i *IdentifierInfo) Hover(ctx context.Context, markdownSupported bool, hoverKind HoverKind) (string, error) {
	ctx, ts := trace.StartSpan(ctx, "source.Hover")
	defer ts.End()
	h, err := i.decl.hover(ctx)
	if err != nil {
		return "", err
	}
	var b strings.Builder
	if comment := formatDocumentation(hoverKind, h.comment); comment != "" {
		b.WriteString(comment)
		b.WriteRune('\n')
	}
	if markdownSupported {
		b.WriteString("```go\n")
	}
	switch x := h.source.(type) {
	case ast.Node:
		if err := format.Node(&b, i.File.FileSet(), x); err != nil {
			return "", err
		}
	case types.Object:
		b.WriteString(types.ObjectString(x, i.qf))
	}
	if markdownSupported {
		b.WriteString("\n```")
	}
	return b.String(), nil
}

func formatDocumentation(hoverKind HoverKind, c *ast.CommentGroup) string {
	switch hoverKind {
	case SynopsisDocumentation:
		return doc.Synopsis((c.Text()))
	case FullDocumentation:
		return c.Text()
	}
	return ""
}

func (d declaration) hover(ctx context.Context) (*documentation, error) {
	ctx, ts := trace.StartSpan(ctx, "source.hover")
	defer ts.End()
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

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

	"golang.org/x/tools/internal/telemetry/trace"
	errors "golang.org/x/xerrors"
)

type HoverInformation struct {
	// Signature is the symbol's signature.
	Signature string `json:"signature"`

	// SingleLine is a single line describing the symbol.
	// This is recommended only for use in clients that show a single line for hover.
	SingleLine string `json:"singleLine"`

	// Synopsis is a single sentence synopsis of the symbol's documentation.
	Synopsis string `json:"synopsis"`

	// FullDocumentation is the symbol's full documentation.
	FullDocumentation string `json:"fullDocumentation"`

	source  interface{}
	comment *ast.CommentGroup
}

func (i *IdentifierInfo) Hover(ctx context.Context) (*HoverInformation, error) {
	ctx, done := trace.StartSpan(ctx, "source.Hover")
	defer done()

	h, err := i.Declaration.hover(ctx)
	if err != nil {
		return nil, err
	}
	// Determine the symbol's signature.
	switch x := h.source.(type) {
	case ast.Node:
		var b strings.Builder
		if err := format.Node(&b, i.View.Session().Cache().FileSet(), x); err != nil {
			return nil, err
		}
		h.Signature = b.String()
	case types.Object:
		h.Signature = objectString(x, i.qf)
	}

	// Set the documentation.
	if i.Declaration.obj != nil {
		h.SingleLine = objectString(i.Declaration.obj, i.qf)
	}
	if h.comment != nil {
		h.FullDocumentation = h.comment.Text()
		h.Synopsis = doc.Synopsis(h.FullDocumentation)
	}
	return h, nil
}

// objectString is a wrapper around the types.ObjectString function.
// It handles adding more information to the object string.
func objectString(obj types.Object, qf types.Qualifier) string {
	str := types.ObjectString(obj, qf)
	switch obj := obj.(type) {
	case *types.Const:
		str = fmt.Sprintf("%s = %s", str, obj.Val())
	}
	return str
}

func (d Declaration) hover(ctx context.Context) (*HoverInformation, error) {
	_, done := trace.StartSpan(ctx, "source.hover")
	defer done()

	obj := d.obj
	switch node := d.node.(type) {
	case *ast.ImportSpec:
		return &HoverInformation{source: node}, nil
	case *ast.GenDecl:
		switch obj := obj.(type) {
		case *types.TypeName, *types.Var, *types.Const, *types.Func:
			return formatGenDecl(node, obj, obj.Type())
		}
	case *ast.TypeSpec:
		if obj.Parent() == types.Universe {
			if obj.Name() == "error" {
				return &HoverInformation{source: node}, nil
			}
			return &HoverInformation{source: node.Name}, nil // comments not needed for builtins
		}
	case *ast.FuncDecl:
		switch obj.(type) {
		case *types.Func:
			return &HoverInformation{source: obj, comment: node.Doc}, nil
		case *types.Builtin:
			return &HoverInformation{source: node.Type, comment: node.Doc}, nil
		}
	}
	return &HoverInformation{source: obj}, nil
}

func formatGenDecl(node *ast.GenDecl, obj types.Object, typ types.Type) (*HoverInformation, error) {
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
		return nil, errors.Errorf("no spec for node %v at position %v", node, obj.Pos())
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
			return &HoverInformation{source: spec.Type, comment: spec.Doc}, nil
		} else {
			return &HoverInformation{source: spec, comment: node.Doc}, nil
		}
	case *ast.ValueSpec:
		return &HoverInformation{source: spec, comment: spec.Doc}, nil
	case *ast.ImportSpec:
		return &HoverInformation{source: spec, comment: spec.Doc}, nil
	}
	return nil, errors.Errorf("unable to format spec %v (%T)", spec, spec)
}

func formatVar(node ast.Spec, obj types.Object) (*HoverInformation, error) {
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
					return &HoverInformation{source: obj, comment: field.Doc}, nil
				} else if field.Comment.Text() != "" {
					return &HoverInformation{source: obj, comment: field.Comment}, nil
				}
			}
		}
	}
	// If we weren't able to find documentation for the object.
	return &HoverInformation{source: obj}, nil
}

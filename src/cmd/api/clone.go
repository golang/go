// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"go/ast"
	"log"
	"reflect"
)

const debugClone = false

// TODO(bradfitz): delete this function (and whole file) once
// http://golang.org/issue/4380 is fixed.
func clone(i interface{}) (cloned interface{}) {
	if debugClone {
		defer func() {
			if !reflect.DeepEqual(i, cloned) {
				log.Printf("cloned %T doesn't match: in=%#v out=%#v", i, i, cloned)
			}
		}()
	}
	switch v := i.(type) {
	case nil:
		return nil
	case *ast.File:
		o := &ast.File{
			Doc:      v.Doc, // shallow
			Package:  v.Package,
			Comments: v.Comments, // shallow
			Name:     v.Name,
			Scope:    v.Scope,
		}
		for _, x := range v.Decls {
			o.Decls = append(o.Decls, clone(x).(ast.Decl))
		}
		for _, x := range v.Imports {
			o.Imports = append(o.Imports, clone(x).(*ast.ImportSpec))
		}
		for _, x := range v.Unresolved {
			o.Unresolved = append(o.Unresolved, x)
		}
		return o
	case *ast.GenDecl:
		o := new(ast.GenDecl)
		*o = *v
		o.Specs = nil
		for _, x := range v.Specs {
			o.Specs = append(o.Specs, clone(x).(ast.Spec))
		}
		return o
	case *ast.TypeSpec:
		o := new(ast.TypeSpec)
		*o = *v
		o.Type = cloneExpr(v.Type)
		return o
	case *ast.InterfaceType:
		o := new(ast.InterfaceType)
		*o = *v
		o.Methods = clone(v.Methods).(*ast.FieldList)
		return o
	case *ast.FieldList:
		if v == nil {
			return v
		}
		o := new(ast.FieldList)
		*o = *v
		o.List = nil
		for _, x := range v.List {
			o.List = append(o.List, clone(x).(*ast.Field))
		}
		return o
	case *ast.Field:
		o := &ast.Field{
			Doc:     v.Doc, // shallow
			Type:    cloneExpr(v.Type),
			Tag:     clone(v.Tag).(*ast.BasicLit),
			Comment: v.Comment, // shallow
		}
		for _, x := range v.Names {
			o.Names = append(o.Names, clone(x).(*ast.Ident))
		}
		return o
	case *ast.FuncType:
		if v == nil {
			return v
		}
		return &ast.FuncType{
			Func:    v.Func,
			Params:  clone(v.Params).(*ast.FieldList),
			Results: clone(v.Results).(*ast.FieldList),
		}
	case *ast.FuncDecl:
		if v == nil {
			return v
		}
		return &ast.FuncDecl{
			Recv: clone(v.Recv).(*ast.FieldList),
			Name: v.Name,
			Type: clone(v.Type).(*ast.FuncType),
			Body: v.Body, // shallow
		}
	case *ast.ValueSpec:
		if v == nil {
			return v
		}
		o := &ast.ValueSpec{
			Type: cloneExpr(v.Type),
		}
		for _, x := range v.Names {
			o.Names = append(o.Names, x)
		}
		for _, x := range v.Values {
			o.Values = append(o.Values, cloneExpr(x))
		}
		return o
	case *ast.CallExpr:
		if v == nil {
			return v
		}
		o := &ast.CallExpr{}
		*o = *v
		o.Args = cloneExprs(v.Args)
		o.Fun = cloneExpr(v.Fun)
		return o
	case *ast.SelectorExpr:
		if v == nil {
			return nil
		}
		return &ast.SelectorExpr{
			X:   cloneExpr(v.X),
			Sel: v.Sel,
		}
	case *ast.ArrayType:
		return &ast.ArrayType{
			Lbrack: v.Lbrack,
			Len:    cloneExpr(v.Len),
			Elt:    cloneExpr(v.Elt),
		}
	case *ast.StructType:
		return &ast.StructType{
			Struct:     v.Struct,
			Fields:     clone(v.Fields).(*ast.FieldList),
			Incomplete: v.Incomplete,
		}
	case *ast.StarExpr:
		return &ast.StarExpr{
			Star: v.Star,
			X:    cloneExpr(v.X),
		}
	case *ast.CompositeLit:
		return &ast.CompositeLit{
			Type:   cloneExpr(v.Type),
			Lbrace: v.Lbrace,
			Elts:   cloneExprs(v.Elts),
			Rbrace: v.Rbrace,
		}
	case *ast.UnaryExpr:
		return &ast.UnaryExpr{
			OpPos: v.OpPos,
			Op:    v.Op,
			X:     cloneExpr(v.X),
		}
	case *ast.BinaryExpr:
		return &ast.BinaryExpr{
			OpPos: v.OpPos,
			Op:    v.Op,
			X:     cloneExpr(v.X),
			Y:     cloneExpr(v.Y),
		}
	case *ast.Ellipsis:
		return &ast.Ellipsis{
			Ellipsis: v.Ellipsis,
			Elt:      cloneExpr(v.Elt),
		}
	case *ast.KeyValueExpr:
		return &ast.KeyValueExpr{
			Key:   cloneExpr(v.Key),
			Colon: v.Colon,
			Value: cloneExpr(v.Value),
		}
	case *ast.FuncLit:
		return &ast.FuncLit{
			Type: clone(v.Type).(*ast.FuncType),
			Body: v.Body, // shallow
		}
	case *ast.MapType:
		return &ast.MapType{
			Map:   v.Map,
			Key:   cloneExpr(v.Key),
			Value: cloneExpr(v.Value),
		}
	case *ast.ParenExpr:
		return &ast.ParenExpr{
			Lparen: v.Lparen,
			X:      cloneExpr(v.X),
			Rparen: v.Rparen,
		}
	case *ast.Ident, *ast.BasicLit:
		return v
	case *ast.ImportSpec:
		return &ast.ImportSpec{
			Doc:     v.Doc, // shallow
			Name:    v.Name,
			Path:    clone(v.Path).(*ast.BasicLit),
			Comment: v.Comment, // shallow
			EndPos:  v.EndPos,
		}
	case *ast.ChanType:
		return &ast.ChanType{
			Begin: v.Begin,
			Arrow: v.Arrow,
			Dir:   v.Dir,
			Value: cloneExpr(v.Value),
		}
	case *ast.TypeAssertExpr:
		return &ast.TypeAssertExpr{
			X:    cloneExpr(v.X),
			Type: cloneExpr(v.Type),
		}
	case *ast.IndexExpr:
		return &ast.IndexExpr{
			X:      cloneExpr(v.X),
			Index:  cloneExpr(v.Index),
			Lbrack: v.Lbrack,
			Rbrack: v.Rbrack,
		}
	}
	panic(fmt.Sprintf("Uncloneable type %T", i))
}

func cloneExpr(x ast.Expr) ast.Expr {
	if x == nil {
		return nil
	}
	return clone(x).(ast.Expr)
}

func cloneExprs(x []ast.Expr) []ast.Expr {
	if x == nil {
		return nil
	}
	o := make([]ast.Expr, len(x))
	for i, x := range x {
		o[i] = cloneExpr(x)
	}
	return o
}

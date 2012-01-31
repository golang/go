// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements export filtering of an AST.

package doc

import "go/ast"

// filterIdentList removes unexported names from list in place
// and returns the resulting list.
//
func filterIdentList(list []*ast.Ident) []*ast.Ident {
	j := 0
	for _, x := range list {
		if ast.IsExported(x.Name) {
			list[j] = x
			j++
		}
	}
	return list[0:j]
}

// filterFieldList removes unexported fields (field names) from the field list
// in place and returns true if fields were removed. Anonymous fields are
// recorded with the parent type. filterType is called with the types of
// all remaining fields.
//
func (r *reader) filterFieldList(parent *namedType, fields *ast.FieldList) (removedFields bool) {
	if fields == nil {
		return
	}
	list := fields.List
	j := 0
	for _, field := range list {
		keepField := false
		if n := len(field.Names); n == 0 {
			// anonymous field
			name := r.recordAnonymousField(parent, field.Type)
			if ast.IsExported(name) {
				keepField = true
			}
		} else {
			field.Names = filterIdentList(field.Names)
			if len(field.Names) < n {
				removedFields = true
			}
			if len(field.Names) > 0 {
				keepField = true
			}
		}
		if keepField {
			r.filterType(nil, field.Type)
			list[j] = field
			j++
		}
	}
	if j < len(list) {
		removedFields = true
	}
	fields.List = list[0:j]
	return
}

// filterParamList applies filterType to each parameter type in fields.
//
func (r *reader) filterParamList(fields *ast.FieldList) {
	if fields != nil {
		for _, f := range fields.List {
			r.filterType(nil, f.Type)
		}
	}
}

// filterType strips any unexported struct fields or method types from typ
// in place. If fields (or methods) have been removed, the corresponding
// struct or interface type has the Incomplete field set to true. 
//
func (r *reader) filterType(parent *namedType, typ ast.Expr) {
	switch t := typ.(type) {
	case *ast.Ident:
		// nothing to do
	case *ast.ParenExpr:
		r.filterType(nil, t.X)
	case *ast.ArrayType:
		r.filterType(nil, t.Elt)
	case *ast.StructType:
		if r.filterFieldList(parent, t.Fields) {
			t.Incomplete = true
		}
	case *ast.FuncType:
		r.filterParamList(t.Params)
		r.filterParamList(t.Results)
	case *ast.InterfaceType:
		if r.filterFieldList(parent, t.Methods) {
			t.Incomplete = true
		}
	case *ast.MapType:
		r.filterType(nil, t.Key)
		r.filterType(nil, t.Value)
	case *ast.ChanType:
		r.filterType(nil, t.Value)
	}
}

func (r *reader) filterSpec(spec ast.Spec) bool {
	switch s := spec.(type) {
	case *ast.ImportSpec:
		// always keep imports so we can collect them
		return true
	case *ast.ValueSpec:
		s.Names = filterIdentList(s.Names)
		if len(s.Names) > 0 {
			r.filterType(nil, s.Type)
			return true
		}
	case *ast.TypeSpec:
		if ast.IsExported(s.Name.Name) {
			r.filterType(r.lookupType(s.Name.Name), s.Type)
			return true
		}
	}
	return false
}

func (r *reader) filterSpecList(list []ast.Spec) []ast.Spec {
	j := 0
	for _, s := range list {
		if r.filterSpec(s) {
			list[j] = s
			j++
		}
	}
	return list[0:j]
}

func (r *reader) filterDecl(decl ast.Decl) bool {
	switch d := decl.(type) {
	case *ast.GenDecl:
		d.Specs = r.filterSpecList(d.Specs)
		return len(d.Specs) > 0
	case *ast.FuncDecl:
		// ok to filter these methods early because any
		// conflicting method will be filtered here, too -
		// thus, removing these methods early will not lead
		// to the false removal of possible conflicts
		return ast.IsExported(d.Name.Name)
	}
	return false
}

// fileExports removes unexported declarations from src in place.
//
func (r *reader) fileExports(src *ast.File) {
	j := 0
	for _, d := range src.Decls {
		if r.filterDecl(d) {
			src.Decls[j] = d
			j++
		}
	}
	src.Decls = src.Decls[0:j]
}

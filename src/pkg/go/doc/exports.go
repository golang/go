// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements export filtering of an AST.

package doc

import "go/ast"

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

func baseName(x ast.Expr) *ast.Ident {
	switch t := x.(type) {
	case *ast.Ident:
		return t
	case *ast.SelectorExpr:
		if _, ok := t.X.(*ast.Ident); ok {
			return t.Sel
		}
	case *ast.StarExpr:
		return baseName(t.X)
	}
	return nil
}

func (doc *docReader) filterFieldList(tinfo *typeInfo, fields *ast.FieldList) (removedFields bool) {
	if fields == nil {
		return false
	}
	list := fields.List
	j := 0
	for _, f := range list {
		keepField := false
		if len(f.Names) == 0 {
			// anonymous field
			name := baseName(f.Type)
			if name != nil && name.IsExported() {
				// we keep the field - in this case doc.addDecl
				// will take care of adding the embedded type
				keepField = true
			} else if tinfo != nil {
				// we don't keep the field - add it as an embedded
				// type so we won't loose its methods, if any
				if embedded := doc.lookupTypeInfo(name.Name); embedded != nil {
					_, ptr := f.Type.(*ast.StarExpr)
					tinfo.addEmbeddedType(embedded, ptr)
				}
			}
		} else {
			n := len(f.Names)
			f.Names = filterIdentList(f.Names)
			if len(f.Names) < n {
				removedFields = true
			}
			keepField = len(f.Names) > 0
		}
		if keepField {
			doc.filterType(nil, f.Type)
			list[j] = f
			j++
		}
	}
	if j < len(list) {
		removedFields = true
	}
	fields.List = list[0:j]
	return
}

func (doc *docReader) filterParamList(fields *ast.FieldList) bool {
	if fields == nil {
		return false
	}
	var b bool
	for _, f := range fields.List {
		if doc.filterType(nil, f.Type) {
			b = true
		}
	}
	return b
}

func (doc *docReader) filterType(tinfo *typeInfo, typ ast.Expr) bool {
	switch t := typ.(type) {
	case *ast.Ident:
		return ast.IsExported(t.Name)
	case *ast.ParenExpr:
		return doc.filterType(nil, t.X)
	case *ast.ArrayType:
		return doc.filterType(nil, t.Elt)
	case *ast.StructType:
		if doc.filterFieldList(tinfo, t.Fields) {
			t.Incomplete = true
		}
		return len(t.Fields.List) > 0
	case *ast.FuncType:
		b1 := doc.filterParamList(t.Params)
		b2 := doc.filterParamList(t.Results)
		return b1 || b2
	case *ast.InterfaceType:
		if doc.filterFieldList(tinfo, t.Methods) {
			t.Incomplete = true
		}
		return len(t.Methods.List) > 0
	case *ast.MapType:
		b1 := doc.filterType(nil, t.Key)
		b2 := doc.filterType(nil, t.Value)
		return b1 || b2
	case *ast.ChanType:
		return doc.filterType(nil, t.Value)
	}
	return false
}

func (doc *docReader) filterSpec(spec ast.Spec) bool {
	switch s := spec.(type) {
	case *ast.ValueSpec:
		s.Names = filterIdentList(s.Names)
		if len(s.Names) > 0 {
			doc.filterType(nil, s.Type)
			return true
		}
	case *ast.TypeSpec:
		if ast.IsExported(s.Name.Name) {
			doc.filterType(doc.lookupTypeInfo(s.Name.Name), s.Type)
			return true
		}
	}
	return false
}

func (doc *docReader) filterSpecList(list []ast.Spec) []ast.Spec {
	j := 0
	for _, s := range list {
		if doc.filterSpec(s) {
			list[j] = s
			j++
		}
	}
	return list[0:j]
}

func (doc *docReader) filterDecl(decl ast.Decl) bool {
	switch d := decl.(type) {
	case *ast.GenDecl:
		d.Specs = doc.filterSpecList(d.Specs)
		return len(d.Specs) > 0
	case *ast.FuncDecl:
		return ast.IsExported(d.Name.Name)
	}
	return false
}

// fileExports trims the AST for a Go file in place such that
// only exported nodes remain. fileExports returns true if
// there are exported declarations; otherwise it returns false.
//
func (doc *docReader) fileExports(src *ast.File) bool {
	j := 0
	for _, d := range src.Decls {
		if doc.filterDecl(d) {
			src.Decls[j] = d
			j++
		}
	}
	src.Decls = src.Decls[0:j]
	return j > 0
}

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package doc

import "go/ast"

type Filter func(string) bool

func matchFields(fields *ast.FieldList, f Filter) bool {
	if fields != nil {
		for _, field := range fields.List {
			for _, name := range field.Names {
				if f(name.Name) {
					return true
				}
			}
		}
	}
	return false
}

func matchDecl(d *ast.GenDecl, f Filter) bool {
	for _, d := range d.Specs {
		switch v := d.(type) {
		case *ast.ValueSpec:
			for _, name := range v.Names {
				if f(name.Name) {
					return true
				}
			}
		case *ast.TypeSpec:
			if f(v.Name.Name) {
				return true
			}
			switch t := v.Type.(type) {
			case *ast.StructType:
				if matchFields(t.Fields, f) {
					return true
				}
			case *ast.InterfaceType:
				if matchFields(t.Methods, f) {
					return true
				}
			}
		}
	}
	return false
}

func filterValueDocs(a []*ValueDoc, f Filter) []*ValueDoc {
	w := 0
	for _, vd := range a {
		if matchDecl(vd.Decl, f) {
			a[w] = vd
			w++
		}
	}
	return a[0:w]
}

func filterFuncDocs(a []*FuncDoc, f Filter) []*FuncDoc {
	w := 0
	for _, fd := range a {
		if f(fd.Name) {
			a[w] = fd
			w++
		}
	}
	return a[0:w]
}

func filterTypeDocs(a []*TypeDoc, f Filter) []*TypeDoc {
	w := 0
	for _, td := range a {
		n := 0 // number of matches
		if matchDecl(td.Decl, f) {
			n = 1
		} else {
			// type name doesn't match, but we may have matching consts, vars, factories or methods
			td.Consts = filterValueDocs(td.Consts, f)
			td.Vars = filterValueDocs(td.Vars, f)
			td.Factories = filterFuncDocs(td.Factories, f)
			td.Methods = filterFuncDocs(td.Methods, f)
			n += len(td.Consts) + len(td.Vars) + len(td.Factories) + len(td.Methods)
		}
		if n > 0 {
			a[w] = td
			w++
		}
	}
	return a[0:w]
}

// Filter eliminates documentation for names that don't pass through the filter f.
// TODO: Recognize "Type.Method" as a name.
//
func (p *PackageDoc) Filter(f Filter) {
	p.Consts = filterValueDocs(p.Consts, f)
	p.Vars = filterValueDocs(p.Vars, f)
	p.Types = filterTypeDocs(p.Types, f)
	p.Funcs = filterFuncDocs(p.Funcs, f)
	p.Doc = "" // don't show top-level package doc
}

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

func filterValues(a []*Value, f Filter) []*Value {
	w := 0
	for _, vd := range a {
		if matchDecl(vd.Decl, f) {
			a[w] = vd
			w++
		}
	}
	return a[0:w]
}

func filterFuncs(a []*Func, f Filter) []*Func {
	w := 0
	for _, fd := range a {
		if f(fd.Name) {
			a[w] = fd
			w++
		}
	}
	return a[0:w]
}

func filterTypes(a []*Type, f Filter) []*Type {
	w := 0
	for _, td := range a {
		n := 0 // number of matches
		if matchDecl(td.Decl, f) {
			n = 1
		} else {
			// type name doesn't match, but we may have matching consts, vars, factories or methods
			td.Consts = filterValues(td.Consts, f)
			td.Vars = filterValues(td.Vars, f)
			td.Funcs = filterFuncs(td.Funcs, f)
			td.Methods = filterFuncs(td.Methods, f)
			n += len(td.Consts) + len(td.Vars) + len(td.Funcs) + len(td.Methods)
		}
		if n > 0 {
			a[w] = td
			w++
		}
	}
	return a[0:w]
}

// Filter eliminates documentation for names that don't pass through the filter f.
// TODO(gri): Recognize "Type.Method" as a name.
//
func (p *Package) Filter(f Filter) {
	p.Consts = filterValues(p.Consts, f)
	p.Vars = filterValues(p.Vars, f)
	p.Types = filterTypes(p.Types, f)
	p.Funcs = filterFuncs(p.Funcs, f)
	p.Doc = "" // don't show top-level package doc
}

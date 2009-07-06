// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ast

import "go/ast"


func filterIdentList(list []*Ident) []*Ident {
	j := 0;
	for _, x := range list {
		if x.IsExported() {
			list[j] = x;
			j++;
		}
	}
	return list[0 : j];
}


func filterType(typ Expr)

func filterFieldList(list []*Field) []*Field {
	j := 0;
	for _, f := range list {
		exported := false;
		if len(f.Names) == 0 {
			// anonymous field
			// TODO(gri) check if the type is exported for anonymous field
			exported = true;
		} else {
			f.Names = filterIdentList(f.Names);
			exported = len(f.Names) > 0;
		}
		if exported {
			filterType(f.Type);
			list[j] = f;
			j++;
		}
	}
	return list[0 : j];
}


func filterType(typ Expr) {
	switch t := typ.(type) {
	case *ArrayType:
		filterType(t.Elt);
	case *StructType:
		t.Fields = filterFieldList(t.Fields);
	case *FuncType:
		t.Params = filterFieldList(t.Params);
		t.Results = filterFieldList(t.Results);
	case *InterfaceType:
		t.Methods = filterFieldList(t.Methods);
	case *MapType:
		filterType(t.Key);
		filterType(t.Value);
	case *ChanType:
		filterType(t.Value);
	}
}


func filterSpec(spec Spec) bool {
	switch s := spec.(type) {
	case *ValueSpec:
		s.Names = filterIdentList(s.Names);
		if len(s.Names) > 0 {
			filterType(s.Type);
			return true;
		}
	case *TypeSpec:
		// TODO(gri) consider stripping forward declarations
		//           of structs, interfaces, functions, and methods
		if s.Name.IsExported() {
			filterType(s.Type);
			return true;
		}
	}
	return false;
}


func filterSpecList(list []Spec) []Spec {
	j := 0;
	for _, s := range list {
		if filterSpec(s) {
			list[j] = s;
			j++;
		}
	}
	return list[0 : j];
}


func filterDecl(decl Decl) bool {
	switch d := decl.(type) {
	case *GenDecl:
		d.Specs = filterSpecList(d.Specs);
		return len(d.Specs) > 0;
	case *FuncDecl:
		// TODO consider removing function declaration altogether if
		//      forward declaration (i.e., if d.Body == nil) because
		//      in that case the actual declaration will come later.
		d.Body = nil;  // strip body
		return d.Name.IsExported();
	}
	return false;
}


// FilterExports trims an AST in place such that only exported nodes remain:
// all top-level identififiers which are not exported and their associated
// information (such as type, initial value, or function body) are removed.
// Non-exported fields and methods of exported types are stripped, and the
// function bodies of exported functions are set to nil.
//
// FilterExports returns true if there is an exported declaration; it returns
// false otherwise.
//
func FilterExports(prog *Program) bool {
	j := 0;
	for _, d := range prog.Decls {
		if filterDecl(d) {
			prog.Decls[j] = d;
			j++;
		}
	}
	prog.Decls = prog.Decls[0 : j];
	prog.Comments = nil;  // remove unassociated comments
	return j > 0;
}

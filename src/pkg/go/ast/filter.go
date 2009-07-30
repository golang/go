// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ast

import (
	"go/ast";
	"go/token";
)


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


// isExportedType assumes that typ is a correct type.
func isExportedType(typ Expr) bool {
	switch t := typ.(type) {
	case *Ident:
		return t.IsExported();
	case *ParenExpr:
		return isExportedType(t.X);
	case *SelectorExpr:
		// assume t.X is a typename
		return t.Sel.IsExported();
	case *StarExpr:
		return isExportedType(t.X);
	}
	return false;
}


func filterType(typ Expr)

func filterFieldList(list []*Field) []*Field {
	j := 0;
	for _, f := range list {
		exported := false;
		if len(f.Names) == 0 {
			// anonymous field
			// (Note that a non-exported anonymous field
			// may still refer to a type with exported
			// fields, so this is not absolutely correct.
			// However, this cannot be done w/o complete
			// type information.)
			exported = isExportedType(f.Type);
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
	if j > 0 && j < len(list) {
		// fields have been stripped but there is at least one left;
		// add a '...' anonymous field instead
		list[j] = &ast.Field{nil, nil, &ast.Ellipsis{}, nil, nil};
		j++;
	}
	return list[0 : j];
}


func filterParamList(list []*Field) {
	for _, f := range list {
		filterType(f.Type);
	}
}


var noPos token.Position;

func filterType(typ Expr) {
	switch t := typ.(type) {
	case *ArrayType:
		filterType(t.Elt);
	case *StructType:
		// don't change if empty struct
		if len(t.Fields) > 0 {
			t.Fields = filterFieldList(t.Fields);
			if len(t.Fields) == 0 {
				// all fields have been stripped - make look like forward-decl
				t.Lbrace = noPos;
				t.Fields = nil;
				t.Rbrace = noPos;
			}
		}
	case *FuncType:
		filterParamList(t.Params);
		filterParamList(t.Results);
	case *InterfaceType:
		// don't change if empty interface
		if len(t.Methods) > 0 {
			t.Methods = filterFieldList(t.Methods);
			if len(t.Methods) == 0 {
				// all methods have been stripped - make look like forward-decl
				t.Lbrace = noPos;
				t.Methods = nil;
				t.Rbrace = noPos;
			}
		}
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
// all top-level identifiers which are not exported and their associated
// information (such as type, initial value, or function body) are removed.
// Non-exported fields and methods of exported types are stripped, and the
// function bodies of exported functions are set to nil.
//
// FilterExports returns true if there is an exported declaration; it returns
// false otherwise.
//
func FilterExports(src *File) bool {
	j := 0;
	for _, d := range src.Decls {
		if filterDecl(d) {
			src.Decls[j] = d;
			j++;
		}
	}
	src.Decls = src.Decls[0 : j];
	return j > 0;
}


// separator is an empty //-style comment that is interspersed between
// different comment groups when they are concatenated into a single group
//
var separator = &Comment{noPos, []byte{'/', '/'}};


// PackageExports returns an AST containing only the exported declarations
// of the package pkg. PackageExports modifies the pkg AST.
//
func PackageExports(pkg *Package) *File {
	// Collect all source files with exported declarations and count
	// the number of package comments and declarations in all files.
	files := make([]*File, len(pkg.Files));
	ncomments := 0;
	ndecls := 0;
	i := 0;
	for _, f := range pkg.Files {
		if f.Doc != nil {
			ncomments += len(f.Doc.List) + 1;  // +1 for separator
		}
		if FilterExports(f) {
			ndecls += len(f.Decls);
			files[i] = f;
			i++;
		}
	}
	files = files[0 : i];

	// Collect package comments from all package files into a single
	// CommentGroup - the collected package documentation. The order
	// is unspecified. In general there should be only one file with
	// a package comment; but it's better to collect extra comments
	// than drop them on the floor.
	var doc *CommentGroup;
	if ncomments > 0 {
		list := make([]*Comment, ncomments - 1);  // -1: no separator before first group
		i := 0;
		for _, f := range pkg.Files {
			if f.Doc != nil {
				if i > 0 {
					// not the first group - add separator
					list[i] = separator;
					i++;
				}
				for _, c := range f.Doc.List {
					list[i] = c;
					i++
				}
			}
		}
		doc = &CommentGroup{list, nil};
	}

	// Collect exported declarations from all package files.
	var decls []Decl;
	if ndecls > 0 {
		decls = make([]Decl, ndecls);
		i := 0;
		for _, f := range files {
			for _, d := range f.Decls {
				decls[i] = d;
				i++;
			}
		}
	}

	return &File{doc, noPos, &Ident{noPos, pkg.Name}, decls, nil};
}

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package docPrinter

import (
	"vector";
	"utf8";
	"unicode";
	"io";
	"fmt";

	"ast";
	"astprinter";
	"template";
)


// ----------------------------------------------------------------------------
// Elementary support

// TODO this should be an AST method
func isExported(name *ast.Ident) bool {
	ch, len := utf8.DecodeRune(name.Lit);
	return unicode.IsUpper(ch);
}


func hasExportedNames(names []*ast.Ident) bool {
	for i, name := range names {
		if isExported(name) {
			return true;
		}
	}
	return false;
}


// ----------------------------------------------------------------------------

type constDoc struct {
	cast *ast.ConstDecl;
}


type varDoc struct {
	vast *ast.VarDecl;
}


type funcDoc struct {
	fast *ast.FuncDecl;
}


type typeDoc struct {
	tast *ast.TypeDecl;
	methods map[string] *funcDoc;
}


type PackageDoc struct {
	name string;  // package name
	imports map[string] string;
	consts map[string] *constDoc;
	types map[string] *typeDoc;
	vars map[string] *varDoc;
	funcs map[string] *funcDoc;
}


// PackageDoc initializes a document to collect package documentation.
// The package name is provided as initial argument. Use AddPackage to
// add the AST for each source file belonging to the same package.
//
func (P *PackageDoc) Init(name string) {
	P.name = name;
	P.imports = make(map[string] string);
	P.consts = make(map[string] *constDoc);
	P.types = make(map[string] *typeDoc);
	P.vars = make(map[string] *varDoc);
	P.funcs = make(map[string] *funcDoc);
}


func (P *PackageDoc) addDecl(decl ast.Decl) {
	switch d := decl.(type) {
	case *ast.ImportDecl:
	case *ast.ConstDecl:
		if hasExportedNames(d.Names) {
		}
	case *ast.TypeDecl:
		if isExported(d.Name) {
		}
	case *ast.VarDecl:
		if hasExportedNames(d.Names) {
		}
	case *ast.FuncDecl:
		if isExported(d.Name) {
			if d.Recv != nil {
				// method
			} else {
				// ordinary function
			}
		}
	case *ast.DeclList:
		for i, decl := range d.List {
			P.addDecl(decl);
		}
	}
}


// AddPackage adds the AST of a source file belonging to the same
// package. The package names must match. If the package was added
// before, AddPackage is a no-op.
//
func (P *PackageDoc) AddPackage(pak *ast.Package) {
	if P.name != string(pak.Name.Lit) {
		panic("package names don't match");
	}
	
	// add all declarations
	for i, decl := range pak.Decls {
		P.addDecl(decl);
	}
}


func (P *PackageDoc) printConsts(p *astPrinter.Printer) {
}


func (P *PackageDoc) printTypes(p *astPrinter.Printer) {
}


func (P *PackageDoc) printVars(p *astPrinter.Printer) {
}


func (P *PackageDoc) printFuncs(p *astPrinter.Printer) {
}


func (P *PackageDoc) printPackage(p *astPrinter.Printer) {
}


// TODO make this a parameter for Init or Print?
var templ = template.NewTemplateOrDie("template.html");

func (P *PackageDoc) Print(writer io.Write) {
	var astp astPrinter.Printer;
	astp.Init(writer, nil, true);
	
	err := templ.Apply(writer, "<!--", template.Substitution {
		"PACKAGE_NAME-->" : func() { fmt.Fprint(writer, P.name); },
		"PACKAGE_COMMENT-->": func() { },
		"PACKAGE_INTERFACE-->" : func() { },
		"PACKAGE_BODY-->" : func() { },
	});
	if err != nil {
		panic("print error - exiting");
	}
}

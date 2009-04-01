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
	decl *ast.ConstDecl;
}


type varDoc struct {
	decl *ast.VarDecl;
}


type funcDoc struct {
	decl *ast.FuncDecl;
}


type typeDoc struct {
	decl *ast.TypeDecl;
	factories map[string] *funcDoc;
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
func (doc *PackageDoc) Init(name string) {
	doc.name = name;
	doc.imports = make(map[string] string);
	doc.consts = make(map[string] *constDoc);
	doc.types = make(map[string] *typeDoc);
	doc.vars = make(map[string] *varDoc);
	doc.funcs = make(map[string] *funcDoc);
}


func baseTypeName(typ ast.Expr) string {
	switch t := typ.(type) {
	case *ast.Ident:
		return string(t.Lit);
	case *ast.StarExpr:
		return baseTypeName(t.X);
	}
	return "";
}


func (doc *PackageDoc) lookupTypeDoc(typ ast.Expr) *typeDoc {
	tdoc, found := doc.types[baseTypeName(typ)];
	if found {
		return tdoc;
	}
	return nil;
}


func (doc *PackageDoc) addFunc(fun *ast.FuncDecl) {
	name := string(fun.Name.Lit);
	fdoc := &funcDoc{fun};
	
	// determine if it should be associated with a type
	var typ *typeDoc;
	if fun.Recv != nil {
		// method
		typ = doc.lookupTypeDoc(fun.Recv.Type);
		if typ != nil {
			typ.methods[name] = fdoc;
			return;
		}
	} else {
		// perhaps a factory function
		// determine result type, if any
		if len(fun.Type.Results) >= 1 {
			res := fun.Type.Results[0];
			if len(res.Names) <= 1 {
				// exactly one (named or anonymous) result type
				typ = doc.lookupTypeDoc(res.Type);
				if typ != nil {
					typ.factories[name] = fdoc;
					return;
				}
			}
		}
	}
	// TODO other heuristics (e.g. name is "NewTypename"?)
	
	// ordinary function
	doc.funcs[name] = fdoc;
}


func (doc *PackageDoc) addDecl(decl ast.Decl) {
	switch d := decl.(type) {
	case *ast.ImportDecl:
	case *ast.ConstDecl:
		if hasExportedNames(d.Names) {
		}

	case *ast.TypeDecl:
		if isExported(d.Name) {
			// TODO only add if not there already - or ignore?
			name := string(d.Name.Lit);
			tdoc := &typeDoc{d, make(map[string] *funcDoc), make(map[string] *funcDoc)};
			doc.types[name] = tdoc;
		}

	case *ast.VarDecl:
		if hasExportedNames(d.Names) {
		}

	case *ast.FuncDecl:
		if isExported(d.Name) {
			doc.addFunc(d);
		}

	case *ast.DeclList:
		for i, decl := range d.List {
			doc.addDecl(decl);
		}
	}
}


// AddProgram adds the AST of a source file belonging to the same
// package. The package names must match. If the package was added
// before, AddPackage is a no-op.
//
func (doc *PackageDoc) AddProgram(pak *ast.Program) {
	if doc.name != string(pak.Name.Lit) {
		panic("package names don't match");
	}
	
	// add all declarations
	for i, decl := range pak.Decls {
		doc.addDecl(decl);
	}
}


// ----------------------------------------------------------------------------
// Printing

func htmlEscape(s string) string {
	var esc string;
	for i := 0; i < len(s); i++ {
		switch s[i] {
		case '<': esc = "&lt;";
		case '&': esc = "&amp;";
		default: continue;
		}
		return s[0 : i] + esc + htmlEscape(s[i+1 : len(s)]);
	}
	return s;
}


// Reduce contiguous sequences of '\t' in a string to a single '\t'.
func untabify(s string) string {
	for i := 0; i < len(s); i++ {
		if s[i] == '\t' {
			j := i;
			for j < len(s) && s[j] == '\t' {
				j++;
			}
			if j-i > 1 {  // more then one tab
				return s[0 : i+1] + untabify(s[j : len(s)]);
			}
		}
	}
	return s;
}


func stripWhiteSpace(s []byte) []byte {
	i, j := 0, len(s);
	for i < len(s) && s[i] <= ' ' {
		i++;
	}
	for j > i && s[j-1] <= ' ' {
		j--
	}
	return s[i : j];
}


func cleanComment(s []byte) []byte {
	switch s[1] {
	case '/': s = s[2 : len(s)-1];
	case '*': s = s[2 : len(s)-2];
	default : panic("illegal comment");
	}
	return stripWhiteSpace(s);
}


func printComment(p *astPrinter.Printer, comment ast.Comments) {
	in_paragraph := false;
	for i, c := range comment {
		s := cleanComment(c.Text);
		if len(s) > 0 {
			if !in_paragraph {
				p.Printf("<p>\n");
				in_paragraph = true;
			}
			p.Printf("%s\n", htmlEscape(untabify(string(s))));
		} else {
			if in_paragraph {
				p.Printf("</p>\n");
				in_paragraph = false;
			}
		}
	}
	if in_paragraph {
		p.Printf("</p>\n");
	}
}


func (c *constDoc) printConsts(p *astPrinter.Printer) {
}


func (f *funcDoc) print(p *astPrinter.Printer, hsize int) {
	d := f.decl;
	if d.Recv != nil {
		p.Printf("<h%d>func (", hsize);
		p.Expr(d.Recv.Type);
		p.Printf(") %s</h%d>\n", d.Name.Lit, hsize);
	} else {
		p.Printf("<h%d>func %s</h%d>\n", hsize, d.Name.Lit, hsize);
	}
	p.Printf("<p><code>");
	p.DoFuncDecl(d);
	p.Printf("</code></p>\n");
	if d.Doc != nil {
		printComment(p, d.Doc);
	}
}


func (t *typeDoc) print(p *astPrinter.Printer) {
	d := t.decl;
	p.Printf("<h2>type %s</h2>\n", string(d.Name.Lit));
	p.Printf("<p><pre>");
	p.DoTypeDecl(d);
	p.Printf("</pre></p>\n");
	if d.Doc != nil {
		printComment(p, d.Doc);
	}
	
	// print associated methods, if any
	for name, m := range t.factories {
		m.print(p, 3);
	}

	for name, m := range t.methods {
		m.print(p, 3);
	}
}


func (v *varDoc) print(p *astPrinter.Printer) {
}


/*
func (P *Printer) Interface(p *ast.Program) {
	P.full = false;
	for i := 0; i < len(p.Decls); i++ {
		switch d := p.Decls[i].(type) {
		case *ast.ConstDecl:
			if hasExportedNames(d.Names) {
				P.Printf("<h2>Constants</h2>\n");
				P.Printf("<p><pre>");
				P.DoConstDecl(d);
				P.String(nopos, "");
				P.Printf("</pre></p>\n");
				if d.Doc != nil {
					P.printComment(d.Doc);
				}
			}

		case *ast.VarDecl:
			if hasExportedNames(d.Names) {
				P.Printf("<h2>Variables</h2>\n");
				P.Printf("<p><pre>");
				P.DoVarDecl(d);
				P.String(nopos, "");
				P.Printf("</pre></p>\n");
				if d.Doc != nil {
					P.printComment(d.Doc);
				}
			}

		case *ast.DeclList:
			
		}
	}
}
*/


// TODO make this a parameter for Init or Print?
var templ = template.NewTemplateOrDie("template.html");

func (doc *PackageDoc) Print(writer io.Write) {
	var p astPrinter.Printer;
	p.Init(writer, nil, true);
	
	// TODO propagate Apply errors
	templ.Apply(writer, "<!--", template.Substitution {
		"PACKAGE_NAME-->" :
			func() {
				fmt.Fprint(writer, doc.name);
			},

		"PROGRAM_HEADER-->":
			func() {
			},

		"CONSTANTS-->" :
			func() {
			},

		"TYPES-->" :
			func() {
				for name, t := range doc.types {
					p.Printf("<hr />\n");
					t.print(&p);
				}
			},

		"VARIABLES-->" :
			func() {
			},

		"FUNCTIONS-->" :
			func() {
				p.Printf("<hr />\n");
				for name, f := range doc.funcs {
					f.print(&p, 2);
				}
			},
	});
}

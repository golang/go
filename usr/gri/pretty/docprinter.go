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
	"token";
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


func hasExportedDecls(decl []ast.Decl) bool {
	for i, d := range decl {
		switch t := d.(type) {
		case *ast.ConstDecl:
			return hasExportedNames(t.Names);
		}
	}
	return false;
}


// ----------------------------------------------------------------------------

type constDoc struct {
	decl *ast.DeclList;
}


type varDoc struct {
	decl *ast.DeclList;
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
	doc ast.Comments;  // package documentation, if any
	consts *vector.Vector;  // list of *ast.DeclList with Tok == token.CONST
	vars *vector.Vector;  // list of *ast.DeclList with Tok == token.CONST
	types map[string] *typeDoc;
	funcs map[string] *funcDoc;
}


// PackageDoc initializes a document to collect package documentation.
// The package name is provided as initial argument. Use AddPackage to
// add the AST for each source file belonging to the same package.
//
func (doc *PackageDoc) Init(name string) {
	doc.name = name;
	doc.consts = vector.New(0);
	doc.types = make(map[string] *typeDoc);
	doc.vars = vector.New(0);
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


func (doc *PackageDoc) addType(typ *ast.TypeDecl) {
	name := string(typ.Name.Lit);
	tdoc := &typeDoc{typ, make(map[string] *funcDoc), make(map[string] *funcDoc)};
	doc.types[name] = tdoc;
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
	case *ast.ConstDecl:
		if hasExportedNames(d.Names) {
			// TODO
		}

	case *ast.TypeDecl:
		if isExported(d.Name) {
			doc.addType(d);
		}

	case *ast.VarDecl:
		if hasExportedNames(d.Names) {
			// TODO
		}

	case *ast.FuncDecl:
		if isExported(d.Name) {
			doc.addFunc(d);
		}

	case *ast.DeclList:
		switch d.Tok {
		case token.IMPORT, token.TYPE:
			for i, decl := range d.List {
				doc.addDecl(decl);
			}
		case token.CONST:
			if hasExportedDecls(d.List) {
				doc.consts.Push(&constDoc{d});
			}
		case token.VAR:
			if hasExportedDecls(d.List) {
				doc.consts.Push(&varDoc{d});
			}
		}
	}
}


// AddProgram adds the AST of a source file belonging to the same
// package. The package names must match. If the source was added
// before, AddProgram is a no-op.
//
func (doc *PackageDoc) AddProgram(prog *ast.Program) {
	if doc.name != string(prog.Name.Lit) {
		panic("package names don't match");
	}

	// add package documentation
	// TODO what to do if there are multiple files?
	if prog.Doc != nil {
		doc.doc = prog.Doc
	}

	// add all declarations
	for i, decl := range prog.Decls {
		doc.addDecl(decl);
	}
}


// ----------------------------------------------------------------------------
// Printing

func htmlEscape(s []byte) []byte {
	var buf io.ByteBuffer;
	
	i0 := 0;
	for i := 0; i < len(s); i++ {
		var esc string;
		switch s[i] {
		case '<': esc = "&lt;";
		case '&': esc = "&amp;";
		default: continue;
		}
		fmt.Fprintf(&buf, "%s%s", s[i0 : i], esc);
		i0 := i+1;  // skip escaped char
	}

	// write the rest
	if i0 > 0 {
		buf.Write(s[i0 : len(s)]);
		s = buf.Data();
	}
	return s;
}


// Reduce contiguous sequences of '\t' in a string to a single '\t'.
// This will produce better results when the string is printed via
// a tabwriter.
// TODO make this functionality optional.
//
func untabify(s []byte) []byte {
	var buf io.ByteBuffer;

	i0 := 0;
	for i := 0; i < len(s); i++ {
		if s[i] == '\t' {
			i++;  // include '\t'
			buf.Write(s[i0 : i]);
			// skip additional tabs
			for i < len(s) && s[i] == '\t' {
				i++;
			}
			i0 := i;
		} else {
			i++;
		}
	}

	// write the rest
	if i0 > 0 {
		buf.Write(s[i0 : len(s)]);
		s = buf.Data();
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


func stripCommentDelimiters(s []byte) []byte {
	switch s[1] {
	case '/': return s[2 : len(s)-1];
	case '*': return s[2 : len(s)-2];
	}
	panic();
	return nil;
}


const /* formatting mode */ (
	in_gap = iota;
	in_paragraph;
	in_preformatted;
)

func printLine(p *astPrinter.Printer, line []byte, mode int) int {
	indented := len(line) > 0 && line[0] == '\t';
	line = stripWhiteSpace(line);
	if len(line) == 0 {
		// empty line
		switch mode {
		case in_paragraph:
			p.Printf("</p>\n");
			mode = in_gap;
		case in_preformatted:
			p.Printf("\n");
			// remain in preformatted
		}
	} else {
		// non-empty line
		if indented {
			switch mode {
			case in_gap:
				p.Printf("<pre>\n");
			case in_paragraph:
				p.Printf("</p>\n");
				p.Printf("<pre>\n");
			}
			mode = in_preformatted;
		} else {
			switch mode {
			case in_gap:
				p.Printf("<p>\n");
			case in_preformatted:
				p.Printf("</pre>\n");
				p.Printf("<p>\n");
			}
			mode = in_paragraph;
		}
		// print line
		p.Printf("%s\n", untabify(htmlEscape(line)));
	}
	return mode;
}


func closeMode(p *astPrinter.Printer, mode int) {
	switch mode {
	case in_paragraph:
		p.Printf("</p>\n");
	case in_preformatted:
		p.Printf("</pre>\n");
	}
}


func printComments(p *astPrinter.Printer, comment ast.Comments) {
	mode := in_gap;
	for i, c := range comment {
		s := stripCommentDelimiters(c.Text);

		// split comment into lines and print the lines
 		i0 := 0;  // beginning of current line
		for i := 0; i < len(s); i++ {
			if s[i] == '\n' {
				// reached line end - print current line
				mode = printLine(p, s[i0 : i], mode);
				i0 = i + 1;  // beginning of next line; skip '\n'
			}
		}

		// print last line
		mode = printLine(p, s[i0 : len(s)], mode);
	}
	closeMode(p, mode);
}


func (c *constDoc) print(p *astPrinter.Printer) {
	printComments(p, c.decl.Doc);
	p.Printf("<pre>");
	p.DoDeclList(c.decl);
	p.Printf("</pre>\n");
}


func (c *varDoc) print(p *astPrinter.Printer) {
	printComments(p, c.decl.Doc);
	p.Printf("<pre>");
	p.DoDeclList(c.decl);
	p.Printf("</pre>\n");
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
	printComments(p, d.Doc);
}


func (t *typeDoc) print(p *astPrinter.Printer) {
	d := t.decl;
	p.Printf("<h2>type %s</h2>\n", string(d.Name.Lit));
	p.Printf("<p><pre>");
	p.DoTypeDecl(d);
	p.Printf("</pre></p>\n");
	printComments(p, d.Doc);
	
	// print associated methods, if any
	for name, m := range t.factories {
		m.print(p, 3);
	}

	for name, m := range t.methods {
		m.print(p, 3);
	}
}


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
				fmt.Fprintf(writer, "<p><code>import \"%s\"</code></p>\n", doc.name);
				printComments(&p, doc.doc);
			},

		"CONSTANTS-->" :
			func() {
				if doc.consts.Len() > 0 {
					fmt.Fprintln(writer, "<hr />");
					fmt.Fprintln(writer, "<h2>Constants</h2>");
					for i := 0; i < doc.consts.Len(); i++ {
						doc.consts.At(i).(*constDoc).print(&p);
					}
				}
			},

		"TYPES-->" :
			func() {
				for name, t := range doc.types {
					fmt.Fprintln(writer, "<hr />");
					t.print(&p);
				}
			},

		"VARIABLES-->" :
			func() {
				if doc.vars.Len() > 0 {
					fmt.Fprintln(writer, "<hr />");
					fmt.Fprintln(writer, "<h2>Variables</h2>");
					for i := 0; i < doc.vars.Len(); i++ {
						doc.vars.At(i).(*varDoc).print(&p);
					}
				}
			},

		"FUNCTIONS-->" :
			func() {
				if len(doc.funcs) > 0 {
					fmt.Fprintln(writer, "<hr />");
					for name, f := range doc.funcs {
						f.print(&p, 2);
					}
				}
			},
	});
}

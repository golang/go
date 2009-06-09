// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package doc

import (
	"container/vector";
	"fmt";
	"go/ast";
	"go/doc";
	"go/token";
	"io";
	"regexp";
	"sort";
	"strings";
)


// ----------------------------------------------------------------------------
// Elementary support

func hasExportedNames(names []*ast.Ident) bool {
	for i, name := range names {
		if name.IsExported() {
			return true;
		}
	}
	return false;
}


func hasExportedSpecs(specs []ast.Spec) bool {
	for i, s := range specs {
		// only called for []astSpec lists of *ast.ValueSpec
		return hasExportedNames(s.(*ast.ValueSpec).Names);
	}
	return false;
}


// ----------------------------------------------------------------------------

type typeDoc struct {
	decl *ast.GenDecl;  // len(decl.Specs) == 1, and the element type is *ast.TypeSpec
	factories map[string] *ast.FuncDecl;
	methods map[string] *ast.FuncDecl;
}


// DocReader accumulates documentation for a single package.
//
type DocReader struct {
	name string;  // package name
	path string;  // import path
	doc ast.Comments;  // package documentation, if any
	consts *vector.Vector;  // list of *ast.GenDecl
	types map[string] *typeDoc;
	vars *vector.Vector;  // list of *ast.GenDecl
	funcs map[string] *ast.FuncDecl;
}


// Init initializes a DocReader to collect package documentation
// for the package with the given package name and import path.
//
func (doc *DocReader) Init(pkg, imp string) {
	doc.name = pkg;
	doc.path = imp;
	doc.consts = vector.New(0);
	doc.types = make(map[string] *typeDoc);
	doc.vars = vector.New(0);
	doc.funcs = make(map[string] *ast.FuncDecl);
}


func baseTypeName(typ ast.Expr) string {
	switch t := typ.(type) {
	case *ast.Ident:
		return string(t.Value);
	case *ast.StarExpr:
		return baseTypeName(t.X);
	}
	return "";
}


func (doc *DocReader) lookupTypeDoc(typ ast.Expr) *typeDoc {
	tdoc, found := doc.types[baseTypeName(typ)];
	if found {
		return tdoc;
	}
	return nil;
}


func (doc *DocReader) addType(decl *ast.GenDecl) {
	typ := decl.Specs[0].(*ast.TypeSpec);
	name := typ.Name.Value;
	tdoc := &typeDoc{decl, make(map[string] *ast.FuncDecl), make(map[string] *ast.FuncDecl)};
	doc.types[name] = tdoc;
}


func (doc *DocReader) addFunc(fun *ast.FuncDecl) {
	name := fun.Name.Value;

	// determine if it should be associated with a type
	var typ *typeDoc;
	if fun.Recv != nil {
		// method
		// (all receiver types must be declared before they are used)
		typ = doc.lookupTypeDoc(fun.Recv.Type);
		if typ != nil {
			// type found (i.e., exported)
			typ.methods[name] = fun;
		}
		// if the type wasn't found, it wasn't exported
		// TODO(gri): a non-exported type may still have exported functions
		//            determine what to do in that case
		return;
	}

	// perhaps a factory function
	// determine result type, if any
	if len(fun.Type.Results) >= 1 {
		res := fun.Type.Results[0];
		if len(res.Names) <= 1 {
			// exactly one (named or anonymous) result type
			typ = doc.lookupTypeDoc(res.Type);
			if typ != nil {
				typ.factories[name] = fun;
				return;
			}
		}
	}

	// ordinary function
	doc.funcs[name] = fun;
}


func (doc *DocReader) addDecl(decl ast.Decl) {
	switch d := decl.(type) {
	case *ast.GenDecl:
		if len(d.Specs) > 0 {
			switch d.Tok {
			case token.IMPORT:
				// ignore
			case token.CONST:
				// constants are always handled as a group
				if hasExportedSpecs(d.Specs) {
					doc.consts.Push(d);
				}
			case token.TYPE:
				// types are handled individually
				for i, spec := range d.Specs {
					s := spec.(*ast.TypeSpec);
					if s.Name.IsExported() {
						// make a (fake) GenDecl node for this TypeSpec
						// (we need to do this here - as opposed to just
						// for printing - so we don't loose the GenDecl
						// documentation)
						var noPos token.Position;
						doc.addType(&ast.GenDecl{d.Doc, d.Pos(), token.TYPE, noPos, []ast.Spec{s}, noPos});
					}
				}
			case token.VAR:
				// variables are always handled as a group
				if hasExportedSpecs(d.Specs) {
					doc.vars.Push(d);
				}
			}
		}
	case *ast.FuncDecl:
		if d.Name.IsExported() {
			doc.addFunc(d);
		}
	}
}


// AddProgram adds the AST for a source file to the DocReader.
// Adding the same AST multiple times is a no-op.
//
func (doc *DocReader) AddProgram(prog *ast.Program) {
	if doc.name != prog.Name.Value {
		panic("package names don't match");
	}

	// add package documentation
	// TODO(gri) what to do if there are multiple files?
	if prog.Doc != nil {
		doc.doc = prog.Doc
	}

	// add all exported declarations
	for i, decl := range prog.Decls {
		doc.addDecl(decl);
	}
}

// ----------------------------------------------------------------------------
// Conversion to external representation

func astComment(comments ast.Comments) string {
	text := make([]string, len(comments));
	for i, c := range comments {
		text[i] = string(c.Text);
	}
	return commentText(text);
}

// ValueDoc is the documentation for a group of declared
// values, either vars or consts.
//
type ValueDoc struct {
	Doc string;
	Decl *ast.GenDecl;
	order int;
}

type sortValueDoc []*ValueDoc
func (p sortValueDoc) Len() int            { return len(p); }
func (p sortValueDoc) Swap(i, j int)       { p[i], p[j] = p[j], p[i]; }


func declName(d *ast.GenDecl) string {
	if len(d.Specs) != 1 {
		return ""
	}

	switch v := d.Specs[0].(type) {
	case *ast.ValueSpec:
		return v.Names[0].Value;
	case *ast.TypeSpec:
		return v.Name.Value;
	}

	return "";
}


func (p sortValueDoc) Less(i, j int) bool {
	// sort by name
	// pull blocks (name = "") up to top
	// in original order
	if ni, nj := declName(p[i].Decl), declName(p[j].Decl); ni != nj {
		return ni < nj;
	}
	return p[i].order < p[j].order;
}


func makeValueDocs(v *vector.Vector) []*ValueDoc {
	d := make([]*ValueDoc, v.Len());
	for i := range d {
		decl := v.At(i).(*ast.GenDecl);
		d[i] = &ValueDoc{astComment(decl.Doc), decl, i};
	}
	sort.Sort(sortValueDoc(d));
	return d;
}


// FuncDoc is the documentation for a func declaration,
// either a top-level function or a method function.
//
type FuncDoc struct {
	Doc string;
	Recv ast.Expr;	// TODO(rsc): Would like string here
	Name string;
	Decl *ast.FuncDecl;
}

type sortFuncDoc []*FuncDoc
func (p sortFuncDoc) Len() int            { return len(p); }
func (p sortFuncDoc) Swap(i, j int)       { p[i], p[j] = p[j], p[i]; }
func (p sortFuncDoc) Less(i, j int) bool  { return p[i].Name < p[j].Name; }


func makeFuncDocs(m map[string] *ast.FuncDecl) []*FuncDoc {
	d := make([]*FuncDoc, len(m));
	i := 0;
	for name, f := range m {
		doc := new(FuncDoc);
		doc.Doc = astComment(f.Doc);
		if f.Recv != nil {
			doc.Recv = f.Recv.Type;
		}
		doc.Name = f.Name.Value;
		doc.Decl = f;
		d[i] = doc;
		i++;
	}
	sort.Sort(sortFuncDoc(d));
	return d;
}


// TypeDoc is the documentation for a declared type.
// Factories is a sorted list of factory functions that return that type.
// Methods is a sorted list of method functions on that type.
type TypeDoc struct {
	Doc string;
	Type *ast.TypeSpec;
	Factories []*FuncDoc;
	Methods []*FuncDoc;
	Decl *ast.GenDecl;
	order int;
}

type sortTypeDoc []*TypeDoc
func (p sortTypeDoc) Len() int            { return len(p); }
func (p sortTypeDoc) Swap(i, j int)       { p[i], p[j] = p[j], p[i]; }
func (p sortTypeDoc) Less(i, j int) bool {
	// sort by name
	// pull blocks (name = "") up to top
	// in original order
	if ni, nj := p[i].Type.Name.Value, p[j].Type.Name.Value; ni != nj {
		return ni < nj;
	}
	return p[i].order < p[j].order;
}


// NOTE(rsc): This would appear not to be correct for type ( )
// blocks, but the doc extractor above has split them into
// individual statements.
func makeTypeDocs(m map[string] *typeDoc) []*TypeDoc {
	d := make([]*TypeDoc, len(m));
	i := 0;
	for name, old := range m {
		typespec := old.decl.Specs[0].(*ast.TypeSpec);
		t := new(TypeDoc);
		t.Doc = astComment(typespec.Doc);
		t.Type = typespec;
		t.Factories = makeFuncDocs(old.factories);
		t.Methods = makeFuncDocs(old.methods);
		t.Decl = old.decl;
		t.order = i;
		d[i] = t;
		i++;
	}
	sort.Sort(sortTypeDoc(d));
	return d;
}


// PackageDoc is the documentation for an entire package.
//
type PackageDoc struct {
	PackageName string;
	ImportPath string;
	Doc string;
	Consts []*ValueDoc;
	Types []*TypeDoc;
	Vars []*ValueDoc;
	Funcs []*FuncDoc;
}


// Doc returns the accumulated documentation for the package.
//
func (doc *DocReader) Doc() *PackageDoc {
	p := new(PackageDoc);
	p.PackageName = doc.name;
	p.ImportPath = doc.path;
	p.Doc = astComment(doc.doc);
	p.Consts = makeValueDocs(doc.consts);
	p.Vars = makeValueDocs(doc.vars);
	p.Types = makeTypeDocs(doc.types);
	p.Funcs = makeFuncDocs(doc.funcs);
	return p;
}


// ----------------------------------------------------------------------------
// Filtering by name

// Does s look like a regular expression?
func isRegexp(s string) bool {
	metachars := ".(|)*+?^$[]";
	for i, c := range s {
		for j, m := range metachars {
			if c == m {
				return true
			}
		}
	}
	return false
}


func match(s string, a []string) bool {
	for i, t := range a {
		if isRegexp(t) {
			if matched, err := regexp.Match(t, s); matched {
				return true;
			}
		}
		if s == t {
			return true;
		}
	}
	return false;
}


func matchDecl(d *ast.GenDecl, names []string) bool {
	for i, d := range d.Specs {
		switch v := d.(type) {
		case *ast.ValueSpec:
			for j, name := range v.Names {
				if match(name.Value, names) {
					return true;
				}
			}
		case *ast.TypeSpec:
			if match(v.Name.Value, names) {
				return true;
			}
		}
	}
	return false;
}


func filterValueDocs(a []*ValueDoc, names []string) []*ValueDoc {
	w := 0;
	for i, vd := range a {
		if matchDecl(vd.Decl, names) {
			a[w] = vd;
			w++;
		}
	}
	return a[0 : w];
}


func filterFuncDocs(a []*FuncDoc, names []string) []*FuncDoc {
	w := 0;
	for i, fd := range a {
		if match(fd.Name, names) {
			a[w] = fd;
			w++;
		}
	}
	return a[0 : w];
}


func filterTypeDocs(a []*TypeDoc, names []string) []*TypeDoc {
	w := 0;
	for i, td := range a {
		match := false;
		if matchDecl(td.Decl, names) {
			match = true;
		} else {
			// type name doesn't match, but we may have matching factories or methods
			td.Factories = filterFuncDocs(td.Factories, names);
			td.Methods = filterFuncDocs(td.Methods, names);
			match = len(td.Factories) > 0 || len(td.Methods) > 0;
		}
		if match {
			a[w] = td;
			w++;
		}
	}
	return a[0 : w];
}


// Filter eliminates information from d that is not
// about one of the given names.
// TODO: Recognize "Type.Method" as a name.
// TODO(r): maybe precompile the regexps.
//
func (p *PackageDoc) Filter(names []string) {
	p.Consts = filterValueDocs(p.Consts, names);
	p.Vars = filterValueDocs(p.Vars, names);
	p.Types = filterTypeDocs(p.Types, names);
	p.Funcs = filterFuncDocs(p.Funcs, names);
	p.Doc = "";	// don't show top-level package doc
}


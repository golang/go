// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package doc

import (
	"container/vector";
	"fmt";
	"go/ast";
	"go/token";
	"io";
	"regexp";
	"sort";
	"strings";
)


// ----------------------------------------------------------------------------

type typeDoc struct {
	// len(decl.Specs) == 1, and the element type is *ast.TypeSpec
	// if the type declaration hasn't been seen yet, decl is nil
	decl *ast.GenDecl;
	// values, factory functions, and methods associated with the type
	values *vector.Vector;  // list of *ast.GenDecl (consts and vars)
	factories map[string] *ast.FuncDecl;
	methods map[string] *ast.FuncDecl;
}


// docReader accumulates documentation for a single package.
// It modifies the AST: Comments (declaration documentation)
// that have been collected by the DocReader are set to nil
// in the respective AST nodes so that they are not printed
// twice (once when printing the documentation and once when
// printing the corresponding AST node).
//
type docReader struct {
	doc *ast.CommentGroup;  // package documentation, if any
	values *vector.Vector;  // list of *ast.GenDecl (consts and vars)
	types map[string] *typeDoc;
	funcs map[string] *ast.FuncDecl;
	bugs *vector.Vector;  // list of *ast.CommentGroup
}


func (doc *docReader) init() {
	doc.values = vector.New(0);
	doc.types = make(map[string] *typeDoc);
	doc.funcs = make(map[string] *ast.FuncDecl);
	doc.bugs = vector.New(0);
}


func (doc *docReader) addType(decl *ast.GenDecl) {
	spec := decl.Specs[0].(*ast.TypeSpec);
	typ := doc.lookupTypeDoc(spec.Name.Value);
	// typ should always be != nil since declared types
	// are always named - be conservative and check
	if typ != nil {
		// a type should be added at most once, so typ.decl
		// should be nil - if it isn't, simply overwrite it
		typ.decl = decl;
	}
}


var predeclaredTypes = map[string]int {
		// basic types
		"bool": 0,
		"byte": 0,
		"int8": 0,
		"int16": 0,
		"int32": 0,
		"int64": 0,
		"uint8": 0,
		"uint16": 0,
		"uint32": 0,
		"uint64": 0,
		"float32": 0,
		"float64": 0,
		"string": 0,
		// convenience types
		"int": 0,
		"uint": 0,
		"uintptr": 0,
		"float": 0,
}


func (doc *docReader) lookupTypeDoc(name string) *typeDoc {
	if name == "" {
		return nil;  // no type docs for anonymous types
	}
	if _, found := predeclaredTypes[name]; found {
		return nil;  // no type docs for predeclared types
	}
	if tdoc, found := doc.types[name]; found {
		return tdoc;
	}
	// type wasn't found - add one without declaration
	tdoc := &typeDoc{nil, vector.New(0), make(map[string] *ast.FuncDecl), make(map[string] *ast.FuncDecl)};
	doc.types[name] = tdoc;
	return tdoc;
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


func (doc *docReader) addValue(decl *ast.GenDecl) {
	// determine if decl should be associated with a type
	// Heuristic: Collect all types and determine the most frequent type.
	//            If it is "dominant enough" the decl is associated with
	//            that type.

	// determine type frequencies
	freq := make(map[string]int);
	prev := "";
	for _, s := range decl.Specs {
		if v, ok := s.(*ast.ValueSpec); ok {
			name := "";
			switch {
			case v.Type != nil:
				// a type is present; determine it's name
				name = baseTypeName(v.Type);
			case decl.Tok == token.CONST:
				// no type is present but we have a constant declaration;
				// use the previous type name (w/o more type information
				// we cannot handle the case of unnamed variables with
				// initializer expressions except for some trivial cases)
				name = prev;
			}
			if name != "" {
				// increase freq count for name
				f := 0;
				if f0, found := freq[name]; found {
					f = f0;
				}
				freq[name] = f+1;
			}
			prev = name;
		}
	}

	// determine most common type
	domName, domFreq := "", 0;
	for name, f := range freq {
		if f > domFreq {
			domName, domFreq = name, f;
		}
	}

	// determine values list
	const threshold = 0.75;
	values := doc.values;
	if domFreq >= int(float(len(decl.Specs)) * threshold) {
		// most common type is "dominant enough"
		typ := doc.lookupTypeDoc(domName);
		if typ != nil {
			values = typ.values;  // associate with that type
		}
	}

	values.Push(decl);
}


func (doc *docReader) addFunc(fun *ast.FuncDecl) {
	name := fun.Name.Value;

	// determine if it should be associated with a type
	if fun.Recv != nil {
		// method
		typ := doc.lookupTypeDoc(baseTypeName(fun.Recv.Type));
		// typ should always be != nil since receiver base
		// types must be named and cannot be predeclared -
		// be conservative and check
		if typ != nil {
			typ.methods[name] = fun;
			return;
		}
	}

	// perhaps a factory function
	// determine result type, if any
	if len(fun.Type.Results) >= 1 {
		res := fun.Type.Results[0];
		if len(res.Names) <= 1 {
			// exactly one (named or anonymous) result type
			typ := doc.lookupTypeDoc(baseTypeName(res.Type));
			if typ != nil {
				// named result type that is not predeclared
				typ.factories[name] = fun;
				return;
			}
		}
	}

	// ordinary function
	// (or method that was not associated to a type for some reason)
	doc.funcs[name] = fun;
}


func (doc *docReader) addDecl(decl ast.Decl) {
	switch d := decl.(type) {
	case *ast.GenDecl:
		if len(d.Specs) > 0 {
			switch d.Tok {
			case token.CONST, token.VAR:
				// constants and variables are always handled as a group
				doc.addValue(d);
			case token.TYPE:
				// types are handled individually
				var noPos token.Position;
				for _, spec := range d.Specs {
					// make a (fake) GenDecl node for this TypeSpec
					// (we need to do this here - as opposed to just
					// for printing - so we don't lose the GenDecl
					// documentation)
					//
					// TODO(gri): Consider just collecting the TypeSpec
					// node (and copy in the GenDecl.doc if there is no
					// doc in the TypeSpec - this is currently done in
					// makeTypeDocs below). Simpler data structures, but
					// would lose GenDecl documentation if the TypeSpec
					// has documentation as well.
					doc.addType(&ast.GenDecl{d.Doc, d.Pos(), token.TYPE, noPos, []ast.Spec{spec}, noPos});
					// A new GenDecl node is created, no need to nil out d.Doc.
				}
			}
		}
	case *ast.FuncDecl:
		doc.addFunc(d);
	}
}


func copyCommentList(list []*ast.Comment) []*ast.Comment {
	copy := make([]*ast.Comment, len(list));
	for i, c := range list {
		copy[i] = c;
	}
	return copy;
}


var (
	// Regexp constructor needs threads - cannot use init expressions
	bug_markers *regexp.Regexp;
	bug_content *regexp.Regexp;
)


// addFile adds the AST for a source file to the docReader.
// Adding the same AST multiple times is a no-op.
//
func (doc *docReader) addFile(src *ast.File) {
	if bug_markers == nil {
		bug_markers = makeRex("^/[/*][ \t]*BUG\\(.*\\):[ \t]*");  // BUG(uid):
		bug_content = makeRex("[^ \n\r\t]+");  // at least one non-whitespace char
	}

	// add package documentation
	if src.Doc != nil {
		// TODO(gri) This won't do the right thing if there is more
		//           than one file with package comments. Consider
		//           using ast.MergePackageFiles which handles these
		//           comments correctly (but currently looses BUG(...)
		//           comments).
		doc.doc = src.Doc;
		src.Doc = nil;  // doc consumed - remove from ast.File node
	}

	// add all declarations
	for _, decl := range src.Decls {
		doc.addDecl(decl);
	}

	// collect BUG(...) comments
	for c := src.Comments; c != nil; c = c.Next {
		text := c.List[0].Text;
		cstr := string(text);
		if m := bug_markers.ExecuteString(cstr); len(m) > 0 {
			// found a BUG comment; maybe empty
			if bstr := cstr[m[1] : len(cstr)]; bug_content.MatchString(bstr) {
				// non-empty BUG comment; collect comment without BUG prefix
				list := copyCommentList(c.List);
				list[0].Text = text[m[1] : len(text)];
				doc.bugs.Push(&ast.CommentGroup{list, nil});
			}
		}
	}
	src.Comments = nil;  // consumed unassociated comments - remove from ast.File node
}


func NewFileDoc(file *ast.File) *PackageDoc {
	var r docReader;
	r.init();
	r.addFile(file);
	return r.newDoc(file.Name.Value, "", "", nil);
}


func NewPackageDoc(pkg *ast.Package, importpath string) *PackageDoc {
	var r docReader;
	r.init();
	filenames := make([]string, len(pkg.Files));
	i := 0;
	for filename, f := range pkg.Files {
		r.addFile(f);
		filenames[i] = filename;
		i++;
	}
	return r.newDoc(pkg.Name, importpath, pkg.Path, filenames);
}


// ----------------------------------------------------------------------------
// Conversion to external representation

func astComment(comment *ast.CommentGroup) string {
	if comment != nil {
		text := make([]string, len(comment.List));
		for i, c := range comment.List {
			text[i] = string(c.Text);
		}
		return commentText(text);
	}
	return "";
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


func makeValueDocs(v *vector.Vector, tok token.Token) []*ValueDoc {
	d := make([]*ValueDoc, v.Len());  // big enough in any case
	n := 0;
	for i := range d {
		decl := v.At(i).(*ast.GenDecl);
		if decl.Tok == tok {
			d[n] = &ValueDoc{astComment(decl.Doc), decl, i};
			n++;
			decl.Doc = nil;  // doc consumed - removed from AST
		}
	}
	d = d[0 : n];
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
	for _, f := range m {
		doc := new(FuncDoc);
		doc.Doc = astComment(f.Doc);
		f.Doc = nil;  // doc consumed - remove from ast.FuncDecl node
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
// Consts and Vars are sorted lists of constants and variables of (mostly) that type.
// Factories is a sorted list of factory functions that return that type.
// Methods is a sorted list of method functions on that type.
type TypeDoc struct {
	Doc string;
	Type *ast.TypeSpec;
	Consts []*ValueDoc;
	Vars []*ValueDoc;
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
// individual declarations.
func (doc *docReader) makeTypeDocs(m map[string] *typeDoc) []*TypeDoc {
	d := make([]*TypeDoc, len(m));
	i := 0;
	for _, old := range m {
		// all typeDocs should have a declaration associated with
		// them after processing an entire package - be conservative
		// and check
		if decl := old.decl; decl != nil {
			typespec := decl.Specs[0].(*ast.TypeSpec);
			t := new(TypeDoc);
			doc := typespec.Doc;
			typespec.Doc = nil;  // doc consumed - remove from ast.TypeSpec node
			if doc == nil {
				// no doc associated with the spec, use the declaration doc, if any
				doc = decl.Doc;
			}
			decl.Doc = nil;  // doc consumed - remove from ast.Decl node
			t.Doc = astComment(doc);
			t.Type = typespec;
			t.Consts = makeValueDocs(old.values, token.CONST);
			t.Vars = makeValueDocs(old.values, token.VAR);
			t.Factories = makeFuncDocs(old.factories);
			t.Methods = makeFuncDocs(old.methods);
			t.Decl = old.decl;
			t.order = i;
			d[i] = t;
			i++;
		} else {
			// no corresponding type declaration found - move any associated
			// values, factory functions, and methods back to the top-level
			// so that they are not lost (this should only happen if a package
			// file containing the explicit type declaration is missing or if
			// an unqualified type name was used after a "." import)
			// 1) move values
			doc.values.AppendVector(old.values);
			// 2) move factory functions
			for name, f := range old.factories {
				doc.funcs[name] = f;
			}
			// 3) move methods
			for name, f := range old.methods {
				// don't overwrite functions with the same name
				if _, found := doc.funcs[name]; !found {
					doc.funcs[name] = f;
				}
			}
		}
	}
	d = d[0 : i];  // some types may have been ignored
	sort.Sort(sortTypeDoc(d));
	return d;
}


func makeBugDocs(v *vector.Vector) []string {
	d := make([]string, v.Len());
	for i := 0; i < v.Len(); i++ {
		d[i] = astComment(v.At(i).(*ast.CommentGroup));
	}
	return d;
}


// PackageDoc is the documentation for an entire package.
//
type PackageDoc struct {
	PackageName string;
	ImportPath string;
	FilePath string;
	Filenames []string;
	Doc string;
	Consts []*ValueDoc;
	Types []*TypeDoc;
	Vars []*ValueDoc;
	Funcs []*FuncDoc;
	Bugs []string;
}


// newDoc returns the accumulated documentation for the package.
//
func (doc *docReader) newDoc(pkgname, importpath, filepath string, filenames []string) *PackageDoc {
	p := new(PackageDoc);
	p.PackageName = pkgname;
	p.ImportPath = importpath;
	p.FilePath = filepath;
	sort.SortStrings(filenames);
	p.Filenames = filenames;
	p.Doc = astComment(doc.doc);
	// makeTypeDocs may extend the list of doc.values and
	// doc.funcs and thus must be called before any other
	// function consuming those lists
	p.Types = doc.makeTypeDocs(doc.types);
	p.Consts = makeValueDocs(doc.values, token.CONST);
	p.Vars = makeValueDocs(doc.values, token.VAR);
	p.Funcs = makeFuncDocs(doc.funcs);
	p.Bugs = makeBugDocs(doc.bugs);
	return p;
}


// ----------------------------------------------------------------------------
// Filtering by name

// Does s look like a regular expression?
func isRegexp(s string) bool {
	metachars := ".(|)*+?^$[]";
	for _, c := range s {
		for _, m := range metachars {
			if c == m {
				return true
			}
		}
	}
	return false
}


func match(s string, a []string) bool {
	for _, t := range a {
		if isRegexp(t) {
			if matched, err := regexp.MatchString(t, s); matched {
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
	for _, d := range d.Specs {
		switch v := d.(type) {
		case *ast.ValueSpec:
			for _, name := range v.Names {
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
	for _, vd := range a {
		if matchDecl(vd.Decl, names) {
			a[w] = vd;
			w++;
		}
	}
	return a[0 : w];
}


func filterFuncDocs(a []*FuncDoc, names []string) []*FuncDoc {
	w := 0;
	for _, fd := range a {
		if match(fd.Name, names) {
			a[w] = fd;
			w++;
		}
	}
	return a[0 : w];
}


func filterTypeDocs(a []*TypeDoc, names []string) []*TypeDoc {
	w := 0;
	for _, td := range a {
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


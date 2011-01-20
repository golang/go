// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The doc package extracts source code documentation from a Go AST.
package doc

import (
	"go/ast"
	"go/token"
	"regexp"
	"sort"
)


// ----------------------------------------------------------------------------

type typeDoc struct {
	// len(decl.Specs) == 1, and the element type is *ast.TypeSpec
	// if the type declaration hasn't been seen yet, decl is nil
	decl *ast.GenDecl
	// values, factory functions, and methods associated with the type
	values    []*ast.GenDecl // consts and vars
	factories map[string]*ast.FuncDecl
	methods   map[string]*ast.FuncDecl
}


// docReader accumulates documentation for a single package.
// It modifies the AST: Comments (declaration documentation)
// that have been collected by the DocReader are set to nil
// in the respective AST nodes so that they are not printed
// twice (once when printing the documentation and once when
// printing the corresponding AST node).
//
type docReader struct {
	doc     *ast.CommentGroup // package documentation, if any
	pkgName string
	values  []*ast.GenDecl // consts and vars
	types   map[string]*typeDoc
	funcs   map[string]*ast.FuncDecl
	bugs    []*ast.CommentGroup
}


func (doc *docReader) init(pkgName string) {
	doc.pkgName = pkgName
	doc.types = make(map[string]*typeDoc)
	doc.funcs = make(map[string]*ast.FuncDecl)
}


func (doc *docReader) addDoc(comments *ast.CommentGroup) {
	if doc.doc == nil {
		// common case: just one package comment
		doc.doc = comments
		return
	}

	// More than one package comment: Usually there will be only
	// one file with a package comment, but it's better to collect
	// all comments than drop them on the floor.
	// (This code isn't particularly clever - no amortized doubling is
	// used - but this situation occurs rarely and is not time-critical.)
	n1 := len(doc.doc.List)
	n2 := len(comments.List)
	list := make([]*ast.Comment, n1+1+n2) // + 1 for separator line
	copy(list, doc.doc.List)
	list[n1] = &ast.Comment{token.NoPos, []byte("//")} // separator line
	copy(list[n1+1:], comments.List)
	doc.doc = &ast.CommentGroup{list}
}


func (doc *docReader) addType(decl *ast.GenDecl) {
	spec := decl.Specs[0].(*ast.TypeSpec)
	typ := doc.lookupTypeDoc(spec.Name.Name)
	// typ should always be != nil since declared types
	// are always named - be conservative and check
	if typ != nil {
		// a type should be added at most once, so typ.decl
		// should be nil - if it isn't, simply overwrite it
		typ.decl = decl
	}
}


func (doc *docReader) lookupTypeDoc(name string) *typeDoc {
	if name == "" {
		return nil // no type docs for anonymous types
	}
	if tdoc, found := doc.types[name]; found {
		return tdoc
	}
	// type wasn't found - add one without declaration
	tdoc := &typeDoc{nil, nil, make(map[string]*ast.FuncDecl), make(map[string]*ast.FuncDecl)}
	doc.types[name] = tdoc
	return tdoc
}


func baseTypeName(typ ast.Expr) string {
	switch t := typ.(type) {
	case *ast.Ident:
		// if the type is not exported, the effect to
		// a client is as if there were no type name
		if t.IsExported() {
			return string(t.Name)
		}
	case *ast.StarExpr:
		return baseTypeName(t.X)
	}
	return ""
}


func (doc *docReader) addValue(decl *ast.GenDecl) {
	// determine if decl should be associated with a type
	// Heuristic: For each typed entry, determine the type name, if any.
	//            If there is exactly one type name that is sufficiently
	//            frequent, associate the decl with the respective type.
	domName := ""
	domFreq := 0
	prev := ""
	for _, s := range decl.Specs {
		if v, ok := s.(*ast.ValueSpec); ok {
			name := ""
			switch {
			case v.Type != nil:
				// a type is present; determine its name
				name = baseTypeName(v.Type)
			case decl.Tok == token.CONST:
				// no type is present but we have a constant declaration;
				// use the previous type name (w/o more type information
				// we cannot handle the case of unnamed variables with
				// initializer expressions except for some trivial cases)
				name = prev
			}
			if name != "" {
				// entry has a named type
				if domName != "" && domName != name {
					// more than one type name - do not associate
					// with any type
					domName = ""
					break
				}
				domName = name
				domFreq++
			}
			prev = name
		}
	}

	// determine values list
	const threshold = 0.75
	values := &doc.values
	if domName != "" && domFreq >= int(float64(len(decl.Specs))*threshold) {
		// typed entries are sufficiently frequent
		typ := doc.lookupTypeDoc(domName)
		if typ != nil {
			values = &typ.values // associate with that type
		}
	}

	*values = append(*values, decl)
}


// Helper function to set the table entry for function f. Makes sure that
// at least one f with associated documentation is stored in table, if there
// are multiple f's with the same name.
func setFunc(table map[string]*ast.FuncDecl, f *ast.FuncDecl) {
	name := f.Name.Name
	if g, exists := table[name]; exists && g.Doc != nil {
		// a function with the same name has already been registered;
		// since it has documentation, assume f is simply another
		// implementation and ignore it
		// TODO(gri) consider collecting all functions, or at least
		//           all comments
		return
	}
	// function doesn't exist or has no documentation; use f
	table[name] = f
}


func (doc *docReader) addFunc(fun *ast.FuncDecl) {
	name := fun.Name.Name

	// determine if it should be associated with a type
	if fun.Recv != nil {
		// method
		typ := doc.lookupTypeDoc(baseTypeName(fun.Recv.List[0].Type))
		if typ != nil {
			// exported receiver type
			setFunc(typ.methods, fun)
		}
		// otherwise don't show the method
		// TODO(gri): There may be exported methods of non-exported types
		// that can be called because of exported values (consts, vars, or
		// function results) of that type. Could determine if that is the
		// case and then show those methods in an appropriate section.
		return
	}

	// perhaps a factory function
	// determine result type, if any
	if fun.Type.Results.NumFields() >= 1 {
		res := fun.Type.Results.List[0]
		if len(res.Names) <= 1 {
			// exactly one (named or anonymous) result associated
			// with the first type in result signature (there may
			// be more than one result)
			tname := baseTypeName(res.Type)
			typ := doc.lookupTypeDoc(tname)
			if typ != nil {
				// named and exported result type

				// Work-around for failure of heuristic: In package os
				// too many functions are considered factory functions
				// for the Error type. Eliminate manually for now as
				// this appears to be the only important case in the
				// current library where the heuristic fails.
				if doc.pkgName == "os" && tname == "Error" &&
					name != "NewError" && name != "NewSyscallError" {
					// not a factory function for os.Error
					setFunc(doc.funcs, fun) // treat as ordinary function
					return
				}

				setFunc(typ.factories, fun)
				return
			}
		}
	}

	// ordinary function
	setFunc(doc.funcs, fun)
}


func (doc *docReader) addDecl(decl ast.Decl) {
	switch d := decl.(type) {
	case *ast.GenDecl:
		if len(d.Specs) > 0 {
			switch d.Tok {
			case token.CONST, token.VAR:
				// constants and variables are always handled as a group
				doc.addValue(d)
			case token.TYPE:
				// types are handled individually
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
					doc.addType(&ast.GenDecl{d.Doc, d.Pos(), token.TYPE, token.NoPos, []ast.Spec{spec}, token.NoPos})
					// A new GenDecl node is created, no need to nil out d.Doc.
				}
			}
		}
	case *ast.FuncDecl:
		doc.addFunc(d)
	}
}


func copyCommentList(list []*ast.Comment) []*ast.Comment {
	return append([]*ast.Comment(nil), list...)
}

var (
	bug_markers = regexp.MustCompile("^/[/*][ \t]*BUG\\(.*\\):[ \t]*") // BUG(uid):
	bug_content = regexp.MustCompile("[^ \n\r\t]+")                    // at least one non-whitespace char
)


// addFile adds the AST for a source file to the docReader.
// Adding the same AST multiple times is a no-op.
//
func (doc *docReader) addFile(src *ast.File) {
	// add package documentation
	if src.Doc != nil {
		doc.addDoc(src.Doc)
		src.Doc = nil // doc consumed - remove from ast.File node
	}

	// add all declarations
	for _, decl := range src.Decls {
		doc.addDecl(decl)
	}

	// collect BUG(...) comments
	for _, c := range src.Comments {
		text := c.List[0].Text
		if m := bug_markers.FindIndex(text); m != nil {
			// found a BUG comment; maybe empty
			if btxt := text[m[1]:]; bug_content.Match(btxt) {
				// non-empty BUG comment; collect comment without BUG prefix
				list := copyCommentList(c.List)
				list[0].Text = text[m[1]:]
				doc.bugs = append(doc.bugs, &ast.CommentGroup{list})
			}
		}
	}
	src.Comments = nil // consumed unassociated comments - remove from ast.File node
}


func NewFileDoc(file *ast.File) *PackageDoc {
	var r docReader
	r.init(file.Name.Name)
	r.addFile(file)
	return r.newDoc("", nil)
}


func NewPackageDoc(pkg *ast.Package, importpath string) *PackageDoc {
	var r docReader
	r.init(pkg.Name)
	filenames := make([]string, len(pkg.Files))
	i := 0
	for filename, f := range pkg.Files {
		r.addFile(f)
		filenames[i] = filename
		i++
	}
	return r.newDoc(importpath, filenames)
}


// ----------------------------------------------------------------------------
// Conversion to external representation

// ValueDoc is the documentation for a group of declared
// values, either vars or consts.
//
type ValueDoc struct {
	Doc   string
	Decl  *ast.GenDecl
	order int
}

type sortValueDoc []*ValueDoc

func (p sortValueDoc) Len() int      { return len(p) }
func (p sortValueDoc) Swap(i, j int) { p[i], p[j] = p[j], p[i] }


func declName(d *ast.GenDecl) string {
	if len(d.Specs) != 1 {
		return ""
	}

	switch v := d.Specs[0].(type) {
	case *ast.ValueSpec:
		return v.Names[0].Name
	case *ast.TypeSpec:
		return v.Name.Name
	}

	return ""
}


func (p sortValueDoc) Less(i, j int) bool {
	// sort by name
	// pull blocks (name = "") up to top
	// in original order
	if ni, nj := declName(p[i].Decl), declName(p[j].Decl); ni != nj {
		return ni < nj
	}
	return p[i].order < p[j].order
}


func makeValueDocs(list []*ast.GenDecl, tok token.Token) []*ValueDoc {
	d := make([]*ValueDoc, len(list)) // big enough in any case
	n := 0
	for i, decl := range list {
		if decl.Tok == tok {
			d[n] = &ValueDoc{CommentText(decl.Doc), decl, i}
			n++
			decl.Doc = nil // doc consumed - removed from AST
		}
	}
	d = d[0:n]
	sort.Sort(sortValueDoc(d))
	return d
}


// FuncDoc is the documentation for a func declaration,
// either a top-level function or a method function.
//
type FuncDoc struct {
	Doc  string
	Recv ast.Expr // TODO(rsc): Would like string here
	Name string
	Decl *ast.FuncDecl
}

type sortFuncDoc []*FuncDoc

func (p sortFuncDoc) Len() int           { return len(p) }
func (p sortFuncDoc) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }
func (p sortFuncDoc) Less(i, j int) bool { return p[i].Name < p[j].Name }


func makeFuncDocs(m map[string]*ast.FuncDecl) []*FuncDoc {
	d := make([]*FuncDoc, len(m))
	i := 0
	for _, f := range m {
		doc := new(FuncDoc)
		doc.Doc = CommentText(f.Doc)
		f.Doc = nil // doc consumed - remove from ast.FuncDecl node
		if f.Recv != nil {
			doc.Recv = f.Recv.List[0].Type
		}
		doc.Name = f.Name.Name
		doc.Decl = f
		d[i] = doc
		i++
	}
	sort.Sort(sortFuncDoc(d))
	return d
}


// TypeDoc is the documentation for a declared type.
// Consts and Vars are sorted lists of constants and variables of (mostly) that type.
// Factories is a sorted list of factory functions that return that type.
// Methods is a sorted list of method functions on that type.
type TypeDoc struct {
	Doc       string
	Type      *ast.TypeSpec
	Consts    []*ValueDoc
	Vars      []*ValueDoc
	Factories []*FuncDoc
	Methods   []*FuncDoc
	Decl      *ast.GenDecl
	order     int
}

type sortTypeDoc []*TypeDoc

func (p sortTypeDoc) Len() int      { return len(p) }
func (p sortTypeDoc) Swap(i, j int) { p[i], p[j] = p[j], p[i] }
func (p sortTypeDoc) Less(i, j int) bool {
	// sort by name
	// pull blocks (name = "") up to top
	// in original order
	if ni, nj := p[i].Type.Name.Name, p[j].Type.Name.Name; ni != nj {
		return ni < nj
	}
	return p[i].order < p[j].order
}


// NOTE(rsc): This would appear not to be correct for type ( )
// blocks, but the doc extractor above has split them into
// individual declarations.
func (doc *docReader) makeTypeDocs(m map[string]*typeDoc) []*TypeDoc {
	d := make([]*TypeDoc, len(m))
	i := 0
	for _, old := range m {
		// all typeDocs should have a declaration associated with
		// them after processing an entire package - be conservative
		// and check
		if decl := old.decl; decl != nil {
			typespec := decl.Specs[0].(*ast.TypeSpec)
			t := new(TypeDoc)
			doc := typespec.Doc
			typespec.Doc = nil // doc consumed - remove from ast.TypeSpec node
			if doc == nil {
				// no doc associated with the spec, use the declaration doc, if any
				doc = decl.Doc
			}
			decl.Doc = nil // doc consumed - remove from ast.Decl node
			t.Doc = CommentText(doc)
			t.Type = typespec
			t.Consts = makeValueDocs(old.values, token.CONST)
			t.Vars = makeValueDocs(old.values, token.VAR)
			t.Factories = makeFuncDocs(old.factories)
			t.Methods = makeFuncDocs(old.methods)
			t.Decl = old.decl
			t.order = i
			d[i] = t
			i++
		} else {
			// no corresponding type declaration found - move any associated
			// values, factory functions, and methods back to the top-level
			// so that they are not lost (this should only happen if a package
			// file containing the explicit type declaration is missing or if
			// an unqualified type name was used after a "." import)
			// 1) move values
			doc.values = append(doc.values, old.values...)
			// 2) move factory functions
			for name, f := range old.factories {
				doc.funcs[name] = f
			}
			// 3) move methods
			for name, f := range old.methods {
				// don't overwrite functions with the same name
				if _, found := doc.funcs[name]; !found {
					doc.funcs[name] = f
				}
			}
		}
	}
	d = d[0:i] // some types may have been ignored
	sort.Sort(sortTypeDoc(d))
	return d
}


func makeBugDocs(list []*ast.CommentGroup) []string {
	d := make([]string, len(list))
	for i, g := range list {
		d[i] = CommentText(g)
	}
	return d
}


// PackageDoc is the documentation for an entire package.
//
type PackageDoc struct {
	PackageName string
	ImportPath  string
	Filenames   []string
	Doc         string
	Consts      []*ValueDoc
	Types       []*TypeDoc
	Vars        []*ValueDoc
	Funcs       []*FuncDoc
	Bugs        []string
}


// newDoc returns the accumulated documentation for the package.
//
func (doc *docReader) newDoc(importpath string, filenames []string) *PackageDoc {
	p := new(PackageDoc)
	p.PackageName = doc.pkgName
	p.ImportPath = importpath
	sort.SortStrings(filenames)
	p.Filenames = filenames
	p.Doc = CommentText(doc.doc)
	// makeTypeDocs may extend the list of doc.values and
	// doc.funcs and thus must be called before any other
	// function consuming those lists
	p.Types = doc.makeTypeDocs(doc.types)
	p.Consts = makeValueDocs(doc.values, token.CONST)
	p.Vars = makeValueDocs(doc.values, token.VAR)
	p.Funcs = makeFuncDocs(doc.funcs)
	p.Bugs = makeBugDocs(doc.bugs)
	return p
}


// ----------------------------------------------------------------------------
// Filtering by name

type Filter func(string) bool


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

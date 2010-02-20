// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The doc package extracts source code documentation from a Go AST.
package doc

import (
	"container/vector"
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
	values    *vector.Vector // list of *ast.GenDecl (consts and vars)
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
	values  *vector.Vector // list of *ast.GenDecl (consts and vars)
	types   map[string]*typeDoc
	funcs   map[string]*ast.FuncDecl
	bugs    *vector.Vector // list of *ast.CommentGroup
}


func (doc *docReader) init(pkgName string) {
	doc.pkgName = pkgName
	doc.values = new(vector.Vector)
	doc.types = make(map[string]*typeDoc)
	doc.funcs = make(map[string]*ast.FuncDecl)
	doc.bugs = new(vector.Vector)
}


func (doc *docReader) addType(decl *ast.GenDecl) {
	spec := decl.Specs[0].(*ast.TypeSpec)
	typ := doc.lookupTypeDoc(spec.Name.Name())
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
	tdoc := &typeDoc{nil, new(vector.Vector), make(map[string]*ast.FuncDecl), make(map[string]*ast.FuncDecl)}
	doc.types[name] = tdoc
	return tdoc
}


func baseTypeName(typ ast.Expr) string {
	switch t := typ.(type) {
	case *ast.Ident:
		// if the type is not exported, the effect to
		// a client is as if there were no type name
		if t.IsExported() {
			return string(t.Name())
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
				// a type is present; determine it's name
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
	values := doc.values
	if domName != "" && domFreq >= int(float(len(decl.Specs))*threshold) {
		// typed entries are sufficiently frequent
		typ := doc.lookupTypeDoc(domName)
		if typ != nil {
			values = typ.values // associate with that type
		}
	}

	values.Push(decl)
}


func (doc *docReader) addFunc(fun *ast.FuncDecl) {
	name := fun.Name.Name()

	// determine if it should be associated with a type
	if fun.Recv != nil {
		// method
		typ := doc.lookupTypeDoc(baseTypeName(fun.Recv.Type))
		if typ != nil {
			// exported receiver type
			typ.methods[name] = fun
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
	if len(fun.Type.Results) >= 1 {
		res := fun.Type.Results[0]
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
					doc.funcs[name] = fun // treat as ordinary function
					return
				}

				typ.factories[name] = fun
				return
			}
		}
	}

	// ordinary function
	doc.funcs[name] = fun
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
				var noPos token.Position
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
					doc.addType(&ast.GenDecl{d.Doc, d.Pos(), token.TYPE, noPos, []ast.Spec{spec}, noPos})
					// A new GenDecl node is created, no need to nil out d.Doc.
				}
			}
		}
	case *ast.FuncDecl:
		doc.addFunc(d)
	}
}


func copyCommentList(list []*ast.Comment) []*ast.Comment {
	copy := make([]*ast.Comment, len(list))
	for i, c := range list {
		copy[i] = c
	}
	return copy
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
		// TODO(gri) This won't do the right thing if there is more
		//           than one file with package comments. Consider
		//           using ast.MergePackageFiles which handles these
		//           comments correctly (but currently looses BUG(...)
		//           comments).
		doc.doc = src.Doc
		src.Doc = nil // doc consumed - remove from ast.File node
	}

	// add all declarations
	for _, decl := range src.Decls {
		doc.addDecl(decl)
	}

	// collect BUG(...) comments
	for _, c := range src.Comments {
		text := c.List[0].Text
		cstr := string(text)
		if m := bug_markers.ExecuteString(cstr); len(m) > 0 {
			// found a BUG comment; maybe empty
			if bstr := cstr[m[1]:]; bug_content.MatchString(bstr) {
				// non-empty BUG comment; collect comment without BUG prefix
				list := copyCommentList(c.List)
				list[0].Text = text[m[1]:]
				doc.bugs.Push(&ast.CommentGroup{list})
			}
		}
	}
	src.Comments = nil // consumed unassociated comments - remove from ast.File node
}


func NewFileDoc(file *ast.File) *PackageDoc {
	var r docReader
	r.init(file.Name.Name())
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
		return v.Names[0].Name()
	case *ast.TypeSpec:
		return v.Name.Name()
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


func makeValueDocs(v *vector.Vector, tok token.Token) []*ValueDoc {
	d := make([]*ValueDoc, v.Len()) // big enough in any case
	n := 0
	for i := range d {
		decl := v.At(i).(*ast.GenDecl)
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
			doc.Recv = f.Recv.Type
		}
		doc.Name = f.Name.Name()
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
	if ni, nj := p[i].Type.Name.Name(), p[j].Type.Name.Name(); ni != nj {
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
			doc.values.AppendVector(old.values)
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


func makeBugDocs(v *vector.Vector) []string {
	d := make([]string, v.Len())
	for i := 0; i < v.Len(); i++ {
		d[i] = CommentText(v.At(i).(*ast.CommentGroup))
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

// Does s look like a regular expression?
func isRegexp(s string) bool {
	metachars := ".(|)*+?^$[]"
	for _, c := range s {
		for _, m := range metachars {
			if c == m {
				return true
			}
		}
	}
	return false
}


func match(s string, names []string) bool {
	for _, t := range names {
		if isRegexp(t) {
			if matched, _ := regexp.MatchString(t, s); matched {
				return true
			}
		}
		if s == t {
			return true
		}
	}
	return false
}


func matchDecl(d *ast.GenDecl, names []string) bool {
	for _, d := range d.Specs {
		switch v := d.(type) {
		case *ast.ValueSpec:
			for _, name := range v.Names {
				if match(name.Name(), names) {
					return true
				}
			}
		case *ast.TypeSpec:
			if match(v.Name.Name(), names) {
				return true
			}
		}
	}
	return false
}


func filterValueDocs(a []*ValueDoc, names []string) []*ValueDoc {
	w := 0
	for _, vd := range a {
		if matchDecl(vd.Decl, names) {
			a[w] = vd
			w++
		}
	}
	return a[0:w]
}


func filterFuncDocs(a []*FuncDoc, names []string) []*FuncDoc {
	w := 0
	for _, fd := range a {
		if match(fd.Name, names) {
			a[w] = fd
			w++
		}
	}
	return a[0:w]
}


func filterTypeDocs(a []*TypeDoc, names []string) []*TypeDoc {
	w := 0
	for _, td := range a {
		n := 0 // number of matches
		if matchDecl(td.Decl, names) {
			n = 1
		} else {
			// type name doesn't match, but we may have matching consts, vars, factories or methods
			td.Consts = filterValueDocs(td.Consts, names)
			td.Vars = filterValueDocs(td.Vars, names)
			td.Factories = filterFuncDocs(td.Factories, names)
			td.Methods = filterFuncDocs(td.Methods, names)
			n += len(td.Consts) + len(td.Vars) + len(td.Factories) + len(td.Methods)
		}
		if n > 0 {
			a[w] = td
			w++
		}
	}
	return a[0:w]
}


// Filter eliminates information from d that is not
// about one of the given names.
// TODO: Recognize "Type.Method" as a name.
// TODO(r): maybe precompile the regexps.
//
func (p *PackageDoc) Filter(names []string) {
	p.Consts = filterValueDocs(p.Consts, names)
	p.Vars = filterValueDocs(p.Vars, names)
	p.Types = filterTypeDocs(p.Types, names)
	p.Funcs = filterFuncDocs(p.Funcs, names)
	p.Doc = "" // don't show top-level package doc
}

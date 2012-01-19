// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package doc

import (
	"go/ast"
	"go/token"
	"regexp"
	"sort"
	"strconv"
)

// ----------------------------------------------------------------------------
// Collection of documentation info

// embeddedType describes the type of an anonymous field.
//
type embeddedType struct {
	typ *typeInfo // the corresponding base type
	ptr bool      // if set, the anonymous field type is a pointer
}

type typeInfo struct {
	name     string // base type name
	isStruct bool
	// len(decl.Specs) == 1, and the element type is *ast.TypeSpec
	// if the type declaration hasn't been seen yet, decl is nil
	decl     *ast.GenDecl
	embedded []embeddedType
	forward  *Type // forward link to processed type documentation

	// declarations associated with the type
	values    []*ast.GenDecl // consts and vars
	factories map[string]*ast.FuncDecl
	methods   map[string]*ast.FuncDecl
}

func (info *typeInfo) exported() bool {
	return ast.IsExported(info.name)
}

func (info *typeInfo) addEmbeddedType(embedded *typeInfo, isPtr bool) {
	info.embedded = append(info.embedded, embeddedType{embedded, isPtr})
}

// docReader accumulates documentation for a single package.
// It modifies the AST: Comments (declaration documentation)
// that have been collected by the DocReader are set to nil
// in the respective AST nodes so that they are not printed
// twice (once when printing the documentation and once when
// printing the corresponding AST node).
//
type docReader struct {
	doc      *ast.CommentGroup // package documentation, if any
	pkgName  string
	mode     Mode
	imports  map[string]int
	values   []*ast.GenDecl // consts and vars
	types    map[string]*typeInfo
	embedded map[string]*typeInfo // embedded types, possibly not exported
	funcs    map[string]*ast.FuncDecl
	bugs     []*ast.CommentGroup
}

func (doc *docReader) init(pkgName string, mode Mode) {
	doc.pkgName = pkgName
	doc.mode = mode
	doc.imports = make(map[string]int)
	doc.types = make(map[string]*typeInfo)
	doc.embedded = make(map[string]*typeInfo)
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
	blankComment := &ast.Comment{token.NoPos, "//"}
	list := append(doc.doc.List, blankComment)
	doc.doc.List = append(list, comments.List...)
}

func (doc *docReader) lookupTypeInfo(name string) *typeInfo {
	if name == "" || name == "_" {
		return nil // no type docs for anonymous types
	}
	if info, found := doc.types[name]; found {
		return info
	}
	// type wasn't found - add one without declaration
	info := &typeInfo{
		name:      name,
		factories: make(map[string]*ast.FuncDecl),
		methods:   make(map[string]*ast.FuncDecl),
	}
	doc.types[name] = info
	return info
}

func baseTypeName(typ ast.Expr, allTypes bool) string {
	switch t := typ.(type) {
	case *ast.Ident:
		// if the type is not exported, the effect to
		// a client is as if there were no type name
		if t.IsExported() || allTypes {
			return t.Name
		}
	case *ast.StarExpr:
		return baseTypeName(t.X, allTypes)
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
				name = baseTypeName(v.Type, false)
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
		typ := doc.lookupTypeInfo(domName)
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
	// strip function body
	fun.Body = nil

	// determine if it should be associated with a type
	if fun.Recv != nil {
		// method
		recvTypeName := baseTypeName(fun.Recv.List[0].Type, true /* exported or not */ )
		var typ *typeInfo
		if ast.IsExported(recvTypeName) {
			// exported recv type: if not found, add it to doc.types
			typ = doc.lookupTypeInfo(recvTypeName)
		} else {
			// unexported recv type: if not found, do not add it
			// (unexported embedded types are added before this
			// phase, so if the type doesn't exist yet, we don't
			// care about this method)
			typ = doc.types[recvTypeName]
		}
		if typ != nil {
			// exported receiver type
			// associate method with the type
			// (if the type is not exported, it may be embedded
			// somewhere so we need to collect the method anyway)
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
			tname := baseTypeName(res.Type, false)
			typ := doc.lookupTypeInfo(tname)
			if typ != nil {
				// named and exported result type
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
			case token.IMPORT:
				// imports are handled individually
				for _, spec := range d.Specs {
					if import_, err := strconv.Unquote(spec.(*ast.ImportSpec).Path.Value); err == nil {
						doc.imports[import_] = 1
					}
				}
			case token.CONST, token.VAR:
				// constants and variables are always handled as a group
				doc.addValue(d)
			case token.TYPE:
				// types are handled individually
				for _, spec := range d.Specs {
					tspec := spec.(*ast.TypeSpec)
					// add the type to the documentation
					info := doc.lookupTypeInfo(tspec.Name.Name)
					if info == nil {
						continue // no name - ignore the type
					}
					// Make a (fake) GenDecl node for this TypeSpec
					// (we need to do this here - as opposed to just
					// for printing - so we don't lose the GenDecl
					// documentation). Since a new GenDecl node is
					// created, there's no need to nil out d.Doc.
					//
					// TODO(gri): Consider just collecting the TypeSpec
					// node (and copy in the GenDecl.doc if there is no
					// doc in the TypeSpec - this is currently done in
					// makeTypes below). Simpler data structures, but
					// would lose GenDecl documentation if the TypeSpec
					// has documentation as well.
					fake := &ast.GenDecl{d.Doc, d.Pos(), token.TYPE, token.NoPos,
						[]ast.Spec{tspec}, token.NoPos}
					// A type should be added at most once, so info.decl
					// should be nil - if it isn't, simply overwrite it.
					info.decl = fake
					// Look for anonymous fields that might contribute methods.
					var fields *ast.FieldList
					switch typ := spec.(*ast.TypeSpec).Type.(type) {
					case *ast.StructType:
						fields = typ.Fields
						info.isStruct = true
					case *ast.InterfaceType:
						fields = typ.Methods
					}
					if fields != nil {
						for _, field := range fields.List {
							if len(field.Names) == 0 {
								// anonymous field - add corresponding type
								// to the info and collect it in doc
								name := baseTypeName(field.Type, true)
								if embedded := doc.lookupTypeInfo(name); embedded != nil {
									_, ptr := field.Type.(*ast.StarExpr)
									info.addEmbeddedType(embedded, ptr)
								}
							}
						}
					}
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
		if m := bug_markers.FindStringIndex(text); m != nil {
			// found a BUG comment; maybe empty
			if btxt := text[m[1]:]; bug_content.MatchString(btxt) {
				// non-empty BUG comment; collect comment without BUG prefix
				list := copyCommentList(c.List)
				list[0].Text = text[m[1]:]
				doc.bugs = append(doc.bugs, &ast.CommentGroup{list})
			}
		}
	}
	src.Comments = nil // consumed unassociated comments - remove from ast.File node
}

// ----------------------------------------------------------------------------
// Conversion to external representation

func (doc *docReader) makeImports() []string {
	list := make([]string, len(doc.imports))
	i := 0
	for import_ := range doc.imports {
		list[i] = import_
		i++
	}
	sort.Strings(list)
	return list
}

type sortValue []*Value

func (p sortValue) Len() int      { return len(p) }
func (p sortValue) Swap(i, j int) { p[i], p[j] = p[j], p[i] }

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

func (p sortValue) Less(i, j int) bool {
	// sort by name
	// pull blocks (name = "") up to top
	// in original order
	if ni, nj := declName(p[i].Decl), declName(p[j].Decl); ni != nj {
		return ni < nj
	}
	return p[i].order < p[j].order
}

func specNames(specs []ast.Spec) []string {
	names := make([]string, len(specs)) // reasonable estimate
	for _, s := range specs {
		// should always be an *ast.ValueSpec, but be careful
		if s, ok := s.(*ast.ValueSpec); ok {
			for _, ident := range s.Names {
				names = append(names, ident.Name)
			}
		}
	}
	return names
}

func makeValues(list []*ast.GenDecl, tok token.Token) []*Value {
	d := make([]*Value, len(list)) // big enough in any case
	n := 0
	for i, decl := range list {
		if decl.Tok == tok {
			d[n] = &Value{decl.Doc.Text(), specNames(decl.Specs), decl, i}
			n++
			decl.Doc = nil // doc consumed - removed from AST
		}
	}
	d = d[0:n]
	sort.Sort(sortValue(d))
	return d
}

type sortFunc []*Func

func (p sortFunc) Len() int           { return len(p) }
func (p sortFunc) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }
func (p sortFunc) Less(i, j int) bool { return p[i].Name < p[j].Name }

func makeFuncs(m map[string]*ast.FuncDecl) []*Func {
	d := make([]*Func, len(m))
	i := 0
	for _, f := range m {
		doc := new(Func)
		doc.Doc = f.Doc.Text()
		f.Doc = nil // doc consumed - remove from ast.FuncDecl node
		if f.Recv != nil {
			doc.Recv = f.Recv.List[0].Type
		}
		doc.Name = f.Name.Name
		doc.Decl = f
		d[i] = doc
		i++
	}
	sort.Sort(sortFunc(d))
	return d
}

type methodSet map[string]*Func

func (mset methodSet) add(m *Func) {
	if mset[m.Name] == nil {
		mset[m.Name] = m
	}
}

type sortMethod []*Method

func (p sortMethod) Len() int           { return len(p) }
func (p sortMethod) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }
func (p sortMethod) Less(i, j int) bool { return p[i].Func.Name < p[j].Func.Name }

func (mset methodSet) sortedList() []*Method {
	list := make([]*Method, len(mset))
	i := 0
	for _, m := range mset {
		list[i] = &Method{Func: m}
		i++
	}
	sort.Sort(sortMethod(list))
	return list
}

type sortType []*Type

func (p sortType) Len() int      { return len(p) }
func (p sortType) Swap(i, j int) { p[i], p[j] = p[j], p[i] }
func (p sortType) Less(i, j int) bool {
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
func (doc *docReader) makeTypes(m map[string]*typeInfo) []*Type {
	// TODO(gri) Consider computing the embedded method information
	//           before calling makeTypes. Then this function can
	//           be single-phased again. Also, it might simplify some
	//           of the logic.
	//
	// phase 1: associate collected declarations with Types
	list := make([]*Type, len(m))
	i := 0
	for _, old := range m {
		// old typeInfos may not have a declaration associated with them
		// if they are not exported but embedded, or because the package
		// is incomplete.
		if decl := old.decl; decl != nil || !old.exported() {
			// process the type even if not exported so that we have
			// its methods in case they are embedded somewhere
			t := new(Type)
			if decl != nil {
				typespec := decl.Specs[0].(*ast.TypeSpec)
				doc := typespec.Doc
				typespec.Doc = nil // doc consumed - remove from ast.TypeSpec node
				if doc == nil {
					// no doc associated with the spec, use the declaration doc, if any
					doc = decl.Doc
				}
				decl.Doc = nil // doc consumed - remove from ast.Decl node
				t.Doc = doc.Text()
				t.Type = typespec
			}
			t.Consts = makeValues(old.values, token.CONST)
			t.Vars = makeValues(old.values, token.VAR)
			t.Funcs = makeFuncs(old.factories)
			t.methods = makeFuncs(old.methods)
			// The list of embedded types' methods is computed from the list
			// of embedded types, some of which may not have been processed
			// yet (i.e., their forward link is nil) - do this in a 2nd phase.
			// The final list of methods can only be computed after that -
			// do this in a 3rd phase.
			t.Decl = old.decl
			t.order = i
			old.forward = t // old has been processed
			// only add the type to the final type list if it
			// is exported or if we want to see all types
			if old.exported() || doc.mode&AllDecls != 0 {
				list[i] = t
				i++
			}
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
	list = list[0:i] // some types may have been ignored

	// phase 2: collect embedded methods for each processed typeInfo
	for _, old := range m {
		if t := old.forward; t != nil {
			// old has been processed into t; collect embedded
			// methods for t from the list of processed embedded
			// types in old (and thus for which the methods are known)
			if old.isStruct {
				// struct
				t.embedded = make(methodSet)
				collectEmbeddedMethods(t.embedded, old, old.name, false)
			} else {
				// interface
				// TODO(gri) fix this
			}
		}
	}

	// phase 3: compute final method set for each Type
	for _, d := range list {
		if len(d.embedded) > 0 {
			// there are embedded methods - exclude
			// the ones with names conflicting with
			// non-embedded methods
			mset := make(methodSet)
			// top-level methods have priority
			for _, m := range d.methods {
				mset.add(m)
			}
			// add non-conflicting embedded methods
			for _, m := range d.embedded {
				mset.add(m)
			}
			d.Methods = mset.sortedList()
		} else {
			// no embedded methods - convert into a Method list
			d.Methods = make([]*Method, len(d.methods))
			for i, m := range d.methods {
				d.Methods[i] = &Method{Func: m}
			}
		}
	}

	sort.Sort(sortType(list))
	return list
}

// collectEmbeddedMethods collects the embedded methods from all
// processed embedded types found in info in mset. It considers
// embedded types at the most shallow level first so that more
// deeply nested embedded methods with conflicting names are
// excluded.
//
func collectEmbeddedMethods(mset methodSet, info *typeInfo, recvTypeName string, embeddedIsPtr bool) {
	for _, e := range info.embedded {
		if e.typ.forward != nil { // == e was processed
			// Once an embedded type was embedded as a pointer type
			// all embedded types in those types are treated like
			// pointer types for the purpose of the receiver type
			// computation; i.e., embeddedIsPtr is sticky for this
			// embedding hierarchy.
			thisEmbeddedIsPtr := embeddedIsPtr || e.ptr
			for _, m := range e.typ.forward.methods {
				mset.add(customizeRecv(m, thisEmbeddedIsPtr, recvTypeName))
			}
			collectEmbeddedMethods(mset, e.typ, recvTypeName, thisEmbeddedIsPtr)
		}
	}
}

func customizeRecv(m *Func, embeddedIsPtr bool, recvTypeName string) *Func {
	if m == nil || m.Decl == nil || m.Decl.Recv == nil || len(m.Decl.Recv.List) != 1 {
		return m // shouldn't happen, but be safe
	}

	// copy existing receiver field and set new type
	newField := *m.Decl.Recv.List[0]
	_, origRecvIsPtr := newField.Type.(*ast.StarExpr)
	var typ ast.Expr = ast.NewIdent(recvTypeName)
	if !embeddedIsPtr && origRecvIsPtr {
		typ = &ast.StarExpr{token.NoPos, typ}
	}
	newField.Type = typ

	// copy existing receiver field list and set new receiver field
	newFieldList := *m.Decl.Recv
	newFieldList.List = []*ast.Field{&newField}

	// copy existing function declaration and set new receiver field list
	newFuncDecl := *m.Decl
	newFuncDecl.Recv = &newFieldList

	// copy existing function documentation and set new declaration
	newM := *m
	newM.Decl = &newFuncDecl
	newM.Recv = typ

	return &newM
}

func makeBugs(list []*ast.CommentGroup) []string {
	d := make([]string, len(list))
	for i, g := range list {
		d[i] = g.Text()
	}
	return d
}

// newDoc returns the accumulated documentation for the package.
//
func (doc *docReader) newDoc(importpath string, filenames []string) *Package {
	p := new(Package)
	p.Name = doc.pkgName
	p.ImportPath = importpath
	sort.Strings(filenames)
	p.Filenames = filenames
	p.Doc = doc.doc.Text()
	// makeTypes may extend the list of doc.values and
	// doc.funcs and thus must be called before any other
	// function consuming those lists
	p.Types = doc.makeTypes(doc.types)
	p.Imports = doc.makeImports()
	p.Consts = makeValues(doc.values, token.CONST)
	p.Vars = makeValues(doc.values, token.VAR)
	p.Funcs = makeFuncs(doc.funcs)
	p.Bugs = makeBugs(doc.bugs)
	return p
}

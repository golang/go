// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package doc

import (
	"cmp"
	"fmt"
	"go/ast"
	"go/token"
	"internal/lazyregexp"
	"path"
	"slices"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"
)

// ----------------------------------------------------------------------------
// function/method sets
//
// Internally, we treat functions like methods and collect them in method sets.

// A methodSet describes a set of methods. Entries where Decl == nil are conflict
// entries (more than one method with the same name at the same embedding level).
type methodSet map[string]*Func

// recvString returns a string representation of recv of the form "T", "*T",
// "T[A, ...]", "*T[A, ...]" or "BADRECV" (if not a proper receiver type).
func recvString(recv ast.Expr) string {
	switch t := recv.(type) {
	case *ast.Ident:
		return t.Name
	case *ast.StarExpr:
		return "*" + recvString(t.X)
	case *ast.IndexExpr:
		// Generic type with one parameter.
		return fmt.Sprintf("%s[%s]", recvString(t.X), recvParam(t.Index))
	case *ast.IndexListExpr:
		// Generic type with multiple parameters.
		if len(t.Indices) > 0 {
			var b strings.Builder
			b.WriteString(recvString(t.X))
			b.WriteByte('[')
			b.WriteString(recvParam(t.Indices[0]))
			for _, e := range t.Indices[1:] {
				b.WriteString(", ")
				b.WriteString(recvParam(e))
			}
			b.WriteByte(']')
			return b.String()
		}
	}
	return "BADRECV"
}

func recvParam(p ast.Expr) string {
	if id, ok := p.(*ast.Ident); ok {
		return id.Name
	}
	return "BADPARAM"
}

// set creates the corresponding Func for f and adds it to mset.
// If there are multiple f's with the same name, set keeps the first
// one with documentation; conflicts are ignored. The boolean
// specifies whether to leave the AST untouched.
func (mset methodSet) set(f *ast.FuncDecl, preserveAST bool) {
	name := f.Name.Name
	if g := mset[name]; g != nil && g.Doc != "" {
		// A function with the same name has already been registered;
		// since it has documentation, assume f is simply another
		// implementation and ignore it. This does not happen if the
		// caller is using go/build.ScanDir to determine the list of
		// files implementing a package.
		return
	}
	// function doesn't exist or has no documentation; use f
	recv := ""
	if f.Recv != nil {
		var typ ast.Expr
		// be careful in case of incorrect ASTs
		if list := f.Recv.List; len(list) == 1 {
			typ = list[0].Type
		}
		recv = recvString(typ)
	}
	mset[name] = &Func{
		Doc:  f.Doc.Text(),
		Name: name,
		Decl: f,
		Recv: recv,
		Orig: recv,
	}
	if !preserveAST {
		f.Doc = nil // doc consumed - remove from AST
	}
}

// add adds method m to the method set; m is ignored if the method set
// already contains a method with the same name at the same or a higher
// level than m.
func (mset methodSet) add(m *Func) {
	old := mset[m.Name]
	if old == nil || m.Level < old.Level {
		mset[m.Name] = m
		return
	}
	if m.Level == old.Level {
		// conflict - mark it using a method with nil Decl
		mset[m.Name] = &Func{
			Name:  m.Name,
			Level: m.Level,
		}
	}
}

// ----------------------------------------------------------------------------
// Named types

// baseTypeName returns the name of the base type of x (or "")
// and whether the type is imported or not.
func baseTypeName(x ast.Expr) (name string, imported bool) {
	switch t := x.(type) {
	case *ast.Ident:
		return t.Name, false
	case *ast.IndexExpr:
		return baseTypeName(t.X)
	case *ast.IndexListExpr:
		return baseTypeName(t.X)
	case *ast.SelectorExpr:
		if _, ok := t.X.(*ast.Ident); ok {
			// only possible for qualified type names;
			// assume type is imported
			return t.Sel.Name, true
		}
	case *ast.ParenExpr:
		return baseTypeName(t.X)
	case *ast.StarExpr:
		return baseTypeName(t.X)
	}
	return "", false
}

// An embeddedSet describes a set of embedded types.
type embeddedSet map[*namedType]bool

// A namedType represents a named unqualified (package local, or possibly
// predeclared) type. The namedType for a type name is always found via
// reader.lookupType.
type namedType struct {
	doc  string       // doc comment for type
	name string       // type name
	decl *ast.GenDecl // nil if declaration hasn't been seen yet

	isEmbedded bool        // true if this type is embedded
	isStruct   bool        // true if this type is a struct
	embedded   embeddedSet // true if the embedded type is a pointer

	// associated declarations
	values  []*Value // consts and vars
	funcs   methodSet
	methods methodSet
}

// ----------------------------------------------------------------------------
// AST reader

// reader accumulates documentation for a single package.
// It modifies the AST: Comments (declaration documentation)
// that have been collected by the reader are set to nil
// in the respective AST nodes so that they are not printed
// twice (once when printing the documentation and once when
// printing the corresponding AST node).
type reader struct {
	mode Mode

	// package properties
	doc       string // package documentation, if any
	filenames []string
	notes     map[string][]*Note

	// imports
	imports      map[string]int
	hasDotImp    bool // if set, package contains a dot import
	importByName map[string]string

	// declarations
	values []*Value // consts and vars
	order  int      // sort order of const and var declarations (when we can't use a name)
	types  map[string]*namedType
	funcs  methodSet

	// support for package-local shadowing of predeclared types
	shadowedPredecl map[string]bool
	fixmap          map[string][]*ast.InterfaceType
}

func (r *reader) isVisible(name string) bool {
	return r.mode&AllDecls != 0 || token.IsExported(name)
}

// lookupType returns the base type with the given name.
// If the base type has not been encountered yet, a new
// type with the given name but no associated declaration
// is added to the type map.
func (r *reader) lookupType(name string) *namedType {
	if name == "" || name == "_" {
		return nil // no type docs for anonymous types
	}
	if typ, found := r.types[name]; found {
		return typ
	}
	// type not found - add one without declaration
	typ := &namedType{
		name:     name,
		embedded: make(embeddedSet),
		funcs:    make(methodSet),
		methods:  make(methodSet),
	}
	r.types[name] = typ
	return typ
}

// recordAnonymousField registers fieldType as the type of an
// anonymous field in the parent type. If the field is imported
// (qualified name) or the parent is nil, the field is ignored.
// The function returns the field name.
func (r *reader) recordAnonymousField(parent *namedType, fieldType ast.Expr) (fname string) {
	fname, imp := baseTypeName(fieldType)
	if parent == nil || imp {
		return
	}
	if ftype := r.lookupType(fname); ftype != nil {
		ftype.isEmbedded = true
		_, ptr := fieldType.(*ast.StarExpr)
		parent.embedded[ftype] = ptr
	}
	return
}

func (r *reader) readDoc(comment *ast.CommentGroup) {
	// By convention there should be only one package comment
	// but collect all of them if there are more than one.
	text := comment.Text()
	if r.doc == "" {
		r.doc = text
		return
	}
	r.doc += "\n" + text
}

func (r *reader) remember(predecl string, typ *ast.InterfaceType) {
	if r.fixmap == nil {
		r.fixmap = make(map[string][]*ast.InterfaceType)
	}
	r.fixmap[predecl] = append(r.fixmap[predecl], typ)
}

func specNames(specs []ast.Spec) []string {
	names := make([]string, 0, len(specs)) // reasonable estimate
	for _, s := range specs {
		// s guaranteed to be an *ast.ValueSpec by readValue
		for _, ident := range s.(*ast.ValueSpec).Names {
			names = append(names, ident.Name)
		}
	}
	return names
}

// readValue processes a const or var declaration.
func (r *reader) readValue(decl *ast.GenDecl) {
	// determine if decl should be associated with a type
	// Heuristic: For each typed entry, determine the type name, if any.
	//            If there is exactly one type name that is sufficiently
	//            frequent, associate the decl with the respective type.
	domName := ""
	domFreq := 0
	prev := ""
	n := 0
	for _, spec := range decl.Specs {
		s, ok := spec.(*ast.ValueSpec)
		if !ok {
			continue // should not happen, but be conservative
		}
		name := ""
		switch {
		case s.Type != nil:
			// a type is present; determine its name
			if n, imp := baseTypeName(s.Type); !imp {
				name = n
			}
		case decl.Tok == token.CONST && len(s.Values) == 0:
			// no type or value is present but we have a constant declaration;
			// use the previous type name (possibly the empty string)
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
		n++
	}

	// nothing to do w/o a legal declaration
	if n == 0 {
		return
	}

	// determine values list with which to associate the Value for this decl
	values := &r.values
	const threshold = 0.75
	if domName != "" && r.isVisible(domName) && domFreq >= int(float64(len(decl.Specs))*threshold) {
		// typed entries are sufficiently frequent
		if typ := r.lookupType(domName); typ != nil {
			values = &typ.values // associate with that type
		}
	}

	*values = append(*values, &Value{
		Doc:   decl.Doc.Text(),
		Names: specNames(decl.Specs),
		Decl:  decl,
		order: r.order,
	})
	if r.mode&PreserveAST == 0 {
		decl.Doc = nil // doc consumed - remove from AST
	}
	// Note: It's important that the order used here is global because the cleanupTypes
	// methods may move values associated with types back into the global list. If the
	// order is list-specific, sorting is not deterministic because the same order value
	// may appear multiple times (was bug, found when fixing #16153).
	r.order++
}

// fields returns a struct's fields or an interface's methods.
func fields(typ ast.Expr) (list []*ast.Field, isStruct bool) {
	var fields *ast.FieldList
	switch t := typ.(type) {
	case *ast.StructType:
		fields = t.Fields
		isStruct = true
	case *ast.InterfaceType:
		fields = t.Methods
	}
	if fields != nil {
		list = fields.List
	}
	return
}

// readType processes a type declaration.
func (r *reader) readType(decl *ast.GenDecl, spec *ast.TypeSpec) {
	typ := r.lookupType(spec.Name.Name)
	if typ == nil {
		return // no name or blank name - ignore the type
	}

	// A type should be added at most once, so typ.decl
	// should be nil - if it is not, simply overwrite it.
	typ.decl = decl

	// compute documentation
	doc := spec.Doc
	if doc == nil {
		// no doc associated with the spec, use the declaration doc, if any
		doc = decl.Doc
	}
	if r.mode&PreserveAST == 0 {
		spec.Doc = nil // doc consumed - remove from AST
		decl.Doc = nil // doc consumed - remove from AST
	}
	typ.doc = doc.Text()

	// record anonymous fields (they may contribute methods)
	// (some fields may have been recorded already when filtering
	// exports, but that's ok)
	var list []*ast.Field
	list, typ.isStruct = fields(spec.Type)
	for _, field := range list {
		if len(field.Names) == 0 {
			r.recordAnonymousField(typ, field.Type)
		}
	}
}

// isPredeclared reports whether n denotes a predeclared type.
func (r *reader) isPredeclared(n string) bool {
	return predeclaredTypes[n] && r.types[n] == nil
}

// readFunc processes a func or method declaration.
func (r *reader) readFunc(fun *ast.FuncDecl) {
	// strip function body if requested.
	if r.mode&PreserveAST == 0 {
		fun.Body = nil
	}

	// associate methods with the receiver type, if any
	if fun.Recv != nil {
		// method
		if len(fun.Recv.List) == 0 {
			// should not happen (incorrect AST); (See issue 17788)
			// don't show this method
			return
		}
		recvTypeName, imp := baseTypeName(fun.Recv.List[0].Type)
		if imp {
			// should not happen (incorrect AST);
			// don't show this method
			return
		}
		if typ := r.lookupType(recvTypeName); typ != nil {
			typ.methods.set(fun, r.mode&PreserveAST != 0)
		}
		// otherwise ignore the method
		// TODO(gri): There may be exported methods of non-exported types
		// that can be called because of exported values (consts, vars, or
		// function results) of that type. Could determine if that is the
		// case and then show those methods in an appropriate section.
		return
	}

	// Associate factory functions with the first visible result type, as long as
	// others are predeclared types.
	if fun.Type.Results.NumFields() >= 1 {
		var typ *namedType // type to associate the function with
		numResultTypes := 0
		for _, res := range fun.Type.Results.List {
			factoryType := res.Type
			if t, ok := factoryType.(*ast.ArrayType); ok {
				// We consider functions that return slices or arrays of type
				// T (or pointers to T) as factory functions of T.
				factoryType = t.Elt
			}
			if n, imp := baseTypeName(factoryType); !imp && r.isVisible(n) && !r.isPredeclared(n) {
				if lookupTypeParam(n, fun.Type.TypeParams) != nil {
					// Issue #49477: don't associate fun with its type parameter result.
					// A type parameter is not a defined type.
					continue
				}
				if t := r.lookupType(n); t != nil {
					typ = t
					numResultTypes++
					if numResultTypes > 1 {
						break
					}
				}
			}
		}
		// If there is exactly one result type,
		// associate the function with that type.
		if numResultTypes == 1 {
			typ.funcs.set(fun, r.mode&PreserveAST != 0)
			return
		}
	}

	// just an ordinary function
	r.funcs.set(fun, r.mode&PreserveAST != 0)
}

// lookupTypeParam searches for type parameters named name within the tparams
// field list, returning the relevant identifier if found, or nil if not.
func lookupTypeParam(name string, tparams *ast.FieldList) *ast.Ident {
	if tparams == nil {
		return nil
	}
	for _, field := range tparams.List {
		for _, id := range field.Names {
			if id.Name == name {
				return id
			}
		}
	}
	return nil
}

var (
	noteMarker    = `([A-Z][A-Z]+)\(([^)]+)\):?`                // MARKER(uid), MARKER at least 2 chars, uid at least 1 char
	noteMarkerRx  = lazyregexp.New(`^[ \t]*` + noteMarker)      // MARKER(uid) at text start
	noteCommentRx = lazyregexp.New(`^/[/*][ \t]*` + noteMarker) // MARKER(uid) at comment start
)

// clean replaces each sequence of space, \r, or \t characters
// with a single space and removes any trailing and leading spaces.
func clean(s string) string {
	var b []byte
	p := byte(' ')
	for i := 0; i < len(s); i++ {
		q := s[i]
		if q == '\r' || q == '\t' {
			q = ' '
		}
		if q != ' ' || p != ' ' {
			b = append(b, q)
			p = q
		}
	}
	// remove trailing blank, if any
	if n := len(b); n > 0 && p == ' ' {
		b = b[0 : n-1]
	}
	return string(b)
}

// readNote collects a single note from a sequence of comments.
func (r *reader) readNote(list []*ast.Comment) {
	text := (&ast.CommentGroup{List: list}).Text()
	if m := noteMarkerRx.FindStringSubmatchIndex(text); m != nil {
		// The note body starts after the marker.
		// We remove any formatting so that we don't
		// get spurious line breaks/indentation when
		// showing the TODO body.
		body := clean(text[m[1]:])
		if body != "" {
			marker := text[m[2]:m[3]]
			r.notes[marker] = append(r.notes[marker], &Note{
				Pos:  list[0].Pos(),
				End:  list[len(list)-1].End(),
				UID:  text[m[4]:m[5]],
				Body: body,
			})
		}
	}
}

// readNotes extracts notes from comments.
// A note must start at the beginning of a comment with "MARKER(uid):"
// and is followed by the note body (e.g., "// BUG(gri): fix this").
// The note ends at the end of the comment group or at the start of
// another note in the same comment group, whichever comes first.
func (r *reader) readNotes(comments []*ast.CommentGroup) {
	for _, group := range comments {
		i := -1 // comment index of most recent note start, valid if >= 0
		list := group.List
		for j, c := range list {
			if noteCommentRx.MatchString(c.Text) {
				if i >= 0 {
					r.readNote(list[i:j])
				}
				i = j
			}
		}
		if i >= 0 {
			r.readNote(list[i:])
		}
	}
}

// readFile adds the AST for a source file to the reader.
func (r *reader) readFile(src *ast.File) {
	// add package documentation
	if src.Doc != nil {
		r.readDoc(src.Doc)
		if r.mode&PreserveAST == 0 {
			src.Doc = nil // doc consumed - remove from AST
		}
	}

	// add all declarations but for functions which are processed in a separate pass
	for _, decl := range src.Decls {
		switch d := decl.(type) {
		case *ast.GenDecl:
			switch d.Tok {
			case token.IMPORT:
				// imports are handled individually
				for _, spec := range d.Specs {
					if s, ok := spec.(*ast.ImportSpec); ok {
						if import_, err := strconv.Unquote(s.Path.Value); err == nil {
							r.imports[import_] = 1
							var name string
							if s.Name != nil {
								name = s.Name.Name
								if name == "." {
									r.hasDotImp = true
								}
							}
							if name != "." {
								if name == "" {
									name = assumedPackageName(import_)
								}
								old, ok := r.importByName[name]
								if !ok {
									r.importByName[name] = import_
								} else if old != import_ && old != "" {
									r.importByName[name] = "" // ambiguous
								}
							}
						}
					}
				}
			case token.CONST, token.VAR:
				// constants and variables are always handled as a group
				r.readValue(d)
			case token.TYPE:
				// types are handled individually
				if len(d.Specs) == 1 && !d.Lparen.IsValid() {
					// common case: single declaration w/o parentheses
					// (if a single declaration is parenthesized,
					// create a new fake declaration below, so that
					// go/doc type declarations always appear w/o
					// parentheses)
					if s, ok := d.Specs[0].(*ast.TypeSpec); ok {
						r.readType(d, s)
					}
					break
				}
				for _, spec := range d.Specs {
					if s, ok := spec.(*ast.TypeSpec); ok {
						// use an individual (possibly fake) declaration
						// for each type; this also ensures that each type
						// gets to (re-)use the declaration documentation
						// if there's none associated with the spec itself
						fake := &ast.GenDecl{
							Doc: d.Doc,
							// don't use the existing TokPos because it
							// will lead to the wrong selection range for
							// the fake declaration if there are more
							// than one type in the group (this affects
							// src/cmd/godoc/godoc.go's posLink_urlFunc)
							TokPos: s.Pos(),
							Tok:    token.TYPE,
							Specs:  []ast.Spec{s},
						}
						r.readType(fake, s)
					}
				}
			}
		}
	}

	// collect MARKER(...): annotations
	r.readNotes(src.Comments)
	if r.mode&PreserveAST == 0 {
		src.Comments = nil // consumed unassociated comments - remove from AST
	}
}

func (r *reader) readPackage(pkg *ast.Package, mode Mode) {
	// initialize reader
	r.filenames = make([]string, len(pkg.Files))
	r.imports = make(map[string]int)
	r.mode = mode
	r.types = make(map[string]*namedType)
	r.funcs = make(methodSet)
	r.notes = make(map[string][]*Note)
	r.importByName = make(map[string]string)

	// sort package files before reading them so that the
	// result does not depend on map iteration order
	i := 0
	for filename := range pkg.Files {
		r.filenames[i] = filename
		i++
	}
	slices.Sort(r.filenames)

	// process files in sorted order
	for _, filename := range r.filenames {
		f := pkg.Files[filename]
		if mode&AllDecls == 0 {
			r.fileExports(f)
		}
		r.readFile(f)
	}

	for name, path := range r.importByName {
		if path == "" {
			delete(r.importByName, name)
		}
	}

	// process functions now that we have better type information
	for _, f := range pkg.Files {
		for _, decl := range f.Decls {
			if d, ok := decl.(*ast.FuncDecl); ok {
				r.readFunc(d)
			}
		}
	}
}

// ----------------------------------------------------------------------------
// Types

func customizeRecv(f *Func, recvTypeName string, embeddedIsPtr bool, level int) *Func {
	if f == nil || f.Decl == nil || f.Decl.Recv == nil || len(f.Decl.Recv.List) != 1 {
		return f // shouldn't happen, but be safe
	}

	// copy existing receiver field and set new type
	newField := *f.Decl.Recv.List[0]
	origPos := newField.Type.Pos()
	_, origRecvIsPtr := newField.Type.(*ast.StarExpr)
	newIdent := &ast.Ident{NamePos: origPos, Name: recvTypeName}
	var typ ast.Expr = newIdent
	if !embeddedIsPtr && origRecvIsPtr {
		newIdent.NamePos++ // '*' is one character
		typ = &ast.StarExpr{Star: origPos, X: newIdent}
	}
	newField.Type = typ

	// copy existing receiver field list and set new receiver field
	newFieldList := *f.Decl.Recv
	newFieldList.List = []*ast.Field{&newField}

	// copy existing function declaration and set new receiver field list
	newFuncDecl := *f.Decl
	newFuncDecl.Recv = &newFieldList

	// copy existing function documentation and set new declaration
	newF := *f
	newF.Decl = &newFuncDecl
	newF.Recv = recvString(typ)
	// the Orig field never changes
	newF.Level = level

	return &newF
}

// collectEmbeddedMethods collects the embedded methods of typ in mset.
func (r *reader) collectEmbeddedMethods(mset methodSet, typ *namedType, recvTypeName string, embeddedIsPtr bool, level int, visited embeddedSet) {
	visited[typ] = true
	for embedded, isPtr := range typ.embedded {
		// Once an embedded type is embedded as a pointer type
		// all embedded types in those types are treated like
		// pointer types for the purpose of the receiver type
		// computation; i.e., embeddedIsPtr is sticky for this
		// embedding hierarchy.
		thisEmbeddedIsPtr := embeddedIsPtr || isPtr
		for _, m := range embedded.methods {
			// only top-level methods are embedded
			if m.Level == 0 {
				mset.add(customizeRecv(m, recvTypeName, thisEmbeddedIsPtr, level))
			}
		}
		if !visited[embedded] {
			r.collectEmbeddedMethods(mset, embedded, recvTypeName, thisEmbeddedIsPtr, level+1, visited)
		}
	}
	delete(visited, typ)
}

// computeMethodSets determines the actual method sets for each type encountered.
func (r *reader) computeMethodSets() {
	for _, t := range r.types {
		// collect embedded methods for t
		if t.isStruct {
			// struct
			r.collectEmbeddedMethods(t.methods, t, t.name, false, 1, make(embeddedSet))
		} else {
			// interface
			// TODO(gri) fix this
		}
	}

	// For any predeclared names that are declared locally, don't treat them as
	// exported fields anymore.
	for predecl := range r.shadowedPredecl {
		for _, ityp := range r.fixmap[predecl] {
			removeAnonymousField(predecl, ityp)
		}
	}
}

// cleanupTypes removes the association of functions and methods with
// types that have no declaration. Instead, these functions and methods
// are shown at the package level. It also removes types with missing
// declarations or which are not visible.
func (r *reader) cleanupTypes() {
	for _, t := range r.types {
		visible := r.isVisible(t.name)
		predeclared := predeclaredTypes[t.name]

		if t.decl == nil && (predeclared || visible && (t.isEmbedded || r.hasDotImp)) {
			// t.name is a predeclared type (and was not redeclared in this package),
			// or it was embedded somewhere but its declaration is missing (because
			// the AST is incomplete), or we have a dot-import (and all bets are off):
			// move any associated values, funcs, and methods back to the top-level so
			// that they are not lost.
			// 1) move values
			r.values = append(r.values, t.values...)
			// 2) move factory functions
			for name, f := range t.funcs {
				// in a correct AST, package-level function names
				// are all different - no need to check for conflicts
				r.funcs[name] = f
			}
			// 3) move methods
			if !predeclared {
				for name, m := range t.methods {
					// don't overwrite functions with the same name - drop them
					if _, found := r.funcs[name]; !found {
						r.funcs[name] = m
					}
				}
			}
		}
		// remove types w/o declaration or which are not visible
		if t.decl == nil || !visible {
			delete(r.types, t.name)
		}
	}
}

// ----------------------------------------------------------------------------
// Sorting

func sortedKeys(m map[string]int) []string {
	list := make([]string, len(m))
	i := 0
	for key := range m {
		list[i] = key
		i++
	}
	slices.Sort(list)
	return list
}

// sortingName returns the name to use when sorting d into place.
func sortingName(d *ast.GenDecl) string {
	if len(d.Specs) == 1 {
		if s, ok := d.Specs[0].(*ast.ValueSpec); ok {
			return s.Names[0].Name
		}
	}
	return ""
}

func sortedValues(m []*Value, tok token.Token) []*Value {
	list := make([]*Value, len(m)) // big enough in any case
	i := 0
	for _, val := range m {
		if val.Decl.Tok == tok {
			list[i] = val
			i++
		}
	}
	list = list[0:i]

	slices.SortFunc(list, func(a, b *Value) int {
		r := strings.Compare(sortingName(a.Decl), sortingName(b.Decl))
		if r != 0 {
			return r
		}
		return cmp.Compare(a.order, b.order)
	})

	return list
}

func sortedTypes(m map[string]*namedType, allMethods bool) []*Type {
	list := make([]*Type, len(m))
	i := 0
	for _, t := range m {
		list[i] = &Type{
			Doc:     t.doc,
			Name:    t.name,
			Decl:    t.decl,
			Consts:  sortedValues(t.values, token.CONST),
			Vars:    sortedValues(t.values, token.VAR),
			Funcs:   sortedFuncs(t.funcs, true),
			Methods: sortedFuncs(t.methods, allMethods),
		}
		i++
	}

	slices.SortFunc(list, func(a, b *Type) int {
		return strings.Compare(a.Name, b.Name)
	})

	return list
}

func removeStar(s string) string {
	if len(s) > 0 && s[0] == '*' {
		return s[1:]
	}
	return s
}

func sortedFuncs(m methodSet, allMethods bool) []*Func {
	list := make([]*Func, len(m))
	i := 0
	for _, m := range m {
		// determine which methods to include
		switch {
		case m.Decl == nil:
			// exclude conflict entry
		case allMethods, m.Level == 0, !token.IsExported(removeStar(m.Orig)):
			// forced inclusion, method not embedded, or method
			// embedded but original receiver type not exported
			list[i] = m
			i++
		}
	}
	list = list[0:i]
	slices.SortFunc(list, func(a, b *Func) int {
		return strings.Compare(a.Name, b.Name)
	})
	return list
}

// noteBodies returns a list of note body strings given a list of notes.
// This is only used to populate the deprecated Package.Bugs field.
func noteBodies(notes []*Note) []string {
	var list []string
	for _, n := range notes {
		list = append(list, n.Body)
	}
	return list
}

// ----------------------------------------------------------------------------
// Predeclared identifiers

// IsPredeclared reports whether s is a predeclared identifier.
func IsPredeclared(s string) bool {
	return predeclaredTypes[s] || predeclaredFuncs[s] || predeclaredConstants[s]
}

var predeclaredTypes = map[string]bool{
	"any":        true,
	"bool":       true,
	"byte":       true,
	"comparable": true,
	"complex64":  true,
	"complex128": true,
	"error":      true,
	"float32":    true,
	"float64":    true,
	"int":        true,
	"int8":       true,
	"int16":      true,
	"int32":      true,
	"int64":      true,
	"rune":       true,
	"string":     true,
	"uint":       true,
	"uint8":      true,
	"uint16":     true,
	"uint32":     true,
	"uint64":     true,
	"uintptr":    true,
}

var predeclaredFuncs = map[string]bool{
	"append":  true,
	"cap":     true,
	"clear":   true,
	"close":   true,
	"complex": true,
	"copy":    true,
	"delete":  true,
	"imag":    true,
	"len":     true,
	"make":    true,
	"max":     true,
	"min":     true,
	"new":     true,
	"panic":   true,
	"print":   true,
	"println": true,
	"real":    true,
	"recover": true,
}

var predeclaredConstants = map[string]bool{
	"false": true,
	"iota":  true,
	"nil":   true,
	"true":  true,
}

// assumedPackageName returns the assumed package name
// for a given import path. This is a copy of
// golang.org/x/tools/internal/imports.ImportPathToAssumedName.
func assumedPackageName(importPath string) string {
	notIdentifier := func(ch rune) bool {
		return !('a' <= ch && ch <= 'z' || 'A' <= ch && ch <= 'Z' ||
			'0' <= ch && ch <= '9' ||
			ch == '_' ||
			ch >= utf8.RuneSelf && (unicode.IsLetter(ch) || unicode.IsDigit(ch)))
	}

	base := path.Base(importPath)
	if strings.HasPrefix(base, "v") {
		if _, err := strconv.Atoi(base[1:]); err == nil {
			dir := path.Dir(importPath)
			if dir != "." {
				base = path.Base(dir)
			}
		}
	}
	base = strings.TrimPrefix(base, "go-")
	if i := strings.IndexFunc(base, notIdentifier); i >= 0 {
		base = base[:i]
	}
	return base
}

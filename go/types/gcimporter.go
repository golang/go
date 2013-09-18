// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements an Importer for gc-generated object files.

package types

import (
	"bufio"
	"errors"
	"fmt"
	"go/ast"
	"go/build"
	"go/token"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"text/scanner"

	"code.google.com/p/go.tools/go/exact"
)

var pkgExts = [...]string{".a", ".5", ".6", ".8"}

// FindPkg returns the filename and unique package id for an import
// path based on package information provided by build.Import (using
// the build.Default build.Context).
// If no file was found, an empty filename is returned.
//
func FindPkg(path, srcDir string) (filename, id string) {
	if len(path) == 0 {
		return
	}

	id = path
	var noext string
	switch {
	default:
		// "x" -> "$GOPATH/pkg/$GOOS_$GOARCH/x.ext", "x"
		// Don't require the source files to be present.
		bp, _ := build.Import(path, srcDir, build.FindOnly|build.AllowBinary)
		if bp.PkgObj == "" {
			return
		}
		noext = strings.TrimSuffix(bp.PkgObj, ".a")

	case build.IsLocalImport(path):
		// "./x" -> "/this/directory/x.ext", "/this/directory/x"
		noext = filepath.Join(srcDir, path)
		id = noext

	case filepath.IsAbs(path):
		// for completeness only - go/build.Import
		// does not support absolute imports
		// "/x" -> "/x.ext", "/x"
		noext = path
	}

	// try extensions
	for _, ext := range pkgExts {
		filename = noext + ext
		if f, err := os.Stat(filename); err == nil && !f.IsDir() {
			return
		}
	}

	filename = "" // not found
	return
}

// GcImportData imports a package by reading the gc-generated export data,
// adds the corresponding package object to the imports map indexed by id,
// and returns the object.
//
// The imports map must contains all packages already imported. The data
// reader position must be the beginning of the export data section. The
// filename is only used in error messages.
//
// If imports[id] contains the completely imported package, that package
// can be used directly, and there is no need to call this function (but
// there is also no harm but for extra time used).
//
func GcImportData(imports map[string]*Package, filename, id string, data *bufio.Reader) (pkg *Package, err error) {
	// support for gcParser error handling
	defer func() {
		switch r := recover().(type) {
		case nil:
			// nothing to do
		case importError:
			err = r
		default:
			panic(r) // internal error
		}
	}()

	var p gcParser
	p.init(filename, id, data, imports)
	pkg = p.parseExport()

	return
}

// GcImport imports a gc-generated package given its import path, adds the
// corresponding package object to the imports map, and returns the object.
// Local import paths are interpreted relative to the current working directory.
// The imports map must contains all packages already imported.
//
func GcImport(imports map[string]*Package, path string) (pkg *Package, err error) {
	if path == "unsafe" {
		return Unsafe, nil
	}

	srcDir := "."
	if build.IsLocalImport(path) {
		srcDir, err = os.Getwd()
		if err != nil {
			return
		}
	}

	filename, id := FindPkg(path, srcDir)
	if filename == "" {
		err = errors.New("can't find import: " + id)
		return
	}

	// no need to re-import if the package was imported completely before
	if pkg = imports[id]; pkg != nil && pkg.complete {
		return
	}

	// open file
	f, err := os.Open(filename)
	if err != nil {
		return
	}
	defer func() {
		f.Close()
		if err != nil {
			// add file name to error
			err = fmt.Errorf("reading export data: %s: %v", filename, err)
		}
	}()

	buf := bufio.NewReader(f)
	if err = FindGcExportData(buf); err != nil {
		return
	}

	pkg, err = GcImportData(imports, filename, id, buf)

	return
}

// ----------------------------------------------------------------------------
// gcParser

// TODO(gri) Imported objects don't have position information.
//           Ideally use the debug table line info; alternatively
//           create some fake position (or the position of the
//           import). That way error messages referring to imported
//           objects can print meaningful information.

// gcParser parses the exports inside a gc compiler-produced
// object/archive file and populates its scope with the results.
type gcParser struct {
	scanner scanner.Scanner
	tok     rune                // current token
	lit     string              // literal string; only valid for Ident, Int, String tokens
	id      string              // package id of imported package
	imports map[string]*Package // package id -> package object
}

func (p *gcParser) init(filename, id string, src io.Reader, imports map[string]*Package) {
	p.scanner.Init(src)
	p.scanner.Error = func(_ *scanner.Scanner, msg string) { p.error(msg) }
	p.scanner.Mode = scanner.ScanIdents | scanner.ScanInts | scanner.ScanChars | scanner.ScanStrings | scanner.ScanComments | scanner.SkipComments
	p.scanner.Whitespace = 1<<'\t' | 1<<' '
	p.scanner.Filename = filename // for good error messages
	p.next()
	p.id = id
	p.imports = imports
	// leave for debugging
	if false {
		// check consistency of imports map
		for _, pkg := range imports {
			if pkg.name == "" {
				fmt.Printf("no package name for %s\n", pkg.Path)
			}
		}
	}
}

func (p *gcParser) next() {
	p.tok = p.scanner.Scan()
	switch p.tok {
	case scanner.Ident, scanner.Int, scanner.Char, scanner.String, '·':
		p.lit = p.scanner.TokenText()
	default:
		p.lit = ""
	}
	// leave for debugging
	if false {
		fmt.Printf("%s: %q -> %q\n", scanner.TokenString(p.tok), p.scanner.TokenText(), p.lit)
	}
}

func declConst(pkg *Package, name string) *Const {
	// the constant may have been imported before - if it exists
	// already in the respective scope, return that constant
	scope := pkg.scope
	if obj := scope.Lookup(name); obj != nil {
		return obj.(*Const)
	}
	// otherwise create a new constant and insert it into the scope
	obj := NewConst(token.NoPos, pkg, name, nil, nil)
	scope.Insert(obj)
	return obj
}

func declTypeName(pkg *Package, name string) *TypeName {
	scope := pkg.scope
	if obj := scope.Lookup(name); obj != nil {
		return obj.(*TypeName)
	}
	obj := NewTypeName(token.NoPos, pkg, name, nil)
	// a named type may be referred to before the underlying type
	// is known - set it up
	obj.typ = &Named{obj: obj}
	scope.Insert(obj)
	return obj
}

func declVar(pkg *Package, name string) *Var {
	scope := pkg.scope
	if obj := scope.Lookup(name); obj != nil {
		return obj.(*Var)
	}
	obj := NewVar(token.NoPos, pkg, name, nil)
	scope.Insert(obj)
	return obj
}

func declFunc(pkg *Package, name string) *Func {
	scope := pkg.scope
	if obj := scope.Lookup(name); obj != nil {
		return obj.(*Func)
	}
	obj := NewFunc(token.NoPos, pkg, name, nil)
	scope.Insert(obj)
	return obj
}

// ----------------------------------------------------------------------------
// Error handling

// Internal errors are boxed as importErrors.
type importError struct {
	pos scanner.Position
	err error
}

func (e importError) Error() string {
	return fmt.Sprintf("import error %s (byte offset = %d): %s", e.pos, e.pos.Offset, e.err)
}

func (p *gcParser) error(err interface{}) {
	if s, ok := err.(string); ok {
		err = errors.New(s)
	}
	// panic with a runtime.Error if err is not an error
	panic(importError{p.scanner.Pos(), err.(error)})
}

func (p *gcParser) errorf(format string, args ...interface{}) {
	p.error(fmt.Sprintf(format, args...))
}

func (p *gcParser) expect(tok rune) string {
	lit := p.lit
	if p.tok != tok {
		p.errorf("expected %s, got %s (%s)", scanner.TokenString(tok), scanner.TokenString(p.tok), lit)
	}
	p.next()
	return lit
}

func (p *gcParser) expectSpecial(tok string) {
	sep := 'x' // not white space
	i := 0
	for i < len(tok) && p.tok == rune(tok[i]) && sep > ' ' {
		sep = p.scanner.Peek() // if sep <= ' ', there is white space before the next token
		p.next()
		i++
	}
	if i < len(tok) {
		p.errorf("expected %q, got %q", tok, tok[0:i])
	}
}

func (p *gcParser) expectKeyword(keyword string) {
	lit := p.expect(scanner.Ident)
	if lit != keyword {
		p.errorf("expected keyword %s, got %q", keyword, lit)
	}
}

// ----------------------------------------------------------------------------
// Qualified and unqualified names

// PackageId = string_lit .
//
func (p *gcParser) parsePackageId() string {
	id, err := strconv.Unquote(p.expect(scanner.String))
	if err != nil {
		p.error(err)
	}
	// id == "" stands for the imported package id
	// (only known at time of package installation)
	if id == "" {
		id = p.id
	}
	return id
}

// PackageName = ident .
//
func (p *gcParser) parsePackageName() string {
	return p.expect(scanner.Ident)
}

// dotIdentifier = ( ident | '·' ) { ident | int | '·' } .
func (p *gcParser) parseDotIdent() string {
	ident := ""
	if p.tok != scanner.Int {
		sep := 'x' // not white space
		for (p.tok == scanner.Ident || p.tok == scanner.Int || p.tok == '·') && sep > ' ' {
			ident += p.lit
			sep = p.scanner.Peek() // if sep <= ' ', there is white space before the next token
			p.next()
		}
	}
	if ident == "" {
		p.expect(scanner.Ident) // use expect() for error handling
	}
	return ident
}

// QualifiedName = "@" PackageId "." dotIdentifier .
//
func (p *gcParser) parseQualifiedName() (id, name string) {
	p.expect('@')
	id = p.parsePackageId()
	p.expect('.')
	name = p.parseDotIdent()
	return
}

// getPkg returns the package for a given id. If the package is
// not found but we have a package name, create the package and
// add it to the p.imports map.
//
func (p *gcParser) getPkg(id, name string) *Package {
	// package unsafe is not in the imports map - handle explicitly
	if id == "unsafe" {
		return Unsafe
	}
	pkg := p.imports[id]
	if pkg == nil && name != "" {
		pkg = NewPackage(id, name, NewScope(nil))
		p.imports[id] = pkg
	}
	return pkg
}

// parseExportedName is like parseQualifiedName, but
// the package id is resolved to an imported *Package.
//
func (p *gcParser) parseExportedName() (pkg *Package, name string) {
	id, name := p.parseQualifiedName()
	pkg = p.getPkg(id, "")
	if pkg == nil {
		p.errorf("%s package not found", id)
	}
	return
}

// ----------------------------------------------------------------------------
// Types

// BasicType = identifier .
//
func (p *gcParser) parseBasicType() Type {
	id := p.expect(scanner.Ident)
	obj := Universe.Lookup(id)
	if obj, ok := obj.(*TypeName); ok {
		return obj.typ
	}
	p.errorf("not a basic type: %s", id)
	return nil
}

// ArrayType = "[" int_lit "]" Type .
//
func (p *gcParser) parseArrayType() Type {
	// "[" already consumed and lookahead known not to be "]"
	lit := p.expect(scanner.Int)
	p.expect(']')
	elt := p.parseType()
	n, err := strconv.ParseInt(lit, 10, 64)
	if err != nil {
		p.error(err)
	}
	return &Array{len: n, elt: elt}
}

// MapType = "map" "[" Type "]" Type .
//
func (p *gcParser) parseMapType() Type {
	p.expectKeyword("map")
	p.expect('[')
	key := p.parseType()
	p.expect(']')
	elt := p.parseType()
	return &Map{key: key, elt: elt}
}

// Name = identifier | "?" | QualifiedName .
//
// If materializePkg is set, the returned package is guaranteed to be set.
// For fully qualified names, the returned package may be a fake package
// (without name, scope, and not in the p.imports map), created for the
// sole purpose of providing a package path. Fake packages are created
// when the package id is not found in the p.imports map; in that case
// we cannot create a real package because we don't have a package name.
// For non-qualified names, the returned package is the imported package.
//
func (p *gcParser) parseName(materializePkg bool) (pkg *Package, name string) {
	switch p.tok {
	case scanner.Ident:
		pkg = p.imports[p.id]
		name = p.lit
		p.next()
	case '?':
		// anonymous
		pkg = p.imports[p.id]
		p.next()
	case '@':
		// exported name prefixed with package path
		var id string
		id, name = p.parseQualifiedName()
		if materializePkg {
			// we don't have a package name - if the package
			// doesn't exist yet, create a fake package instead
			pkg = p.getPkg(id, "")
			if pkg == nil {
				pkg = &Package{path: id}
			}
		}
	default:
		p.error("name expected")
	}
	return
}

// Field = Name Type [ string_lit ] .
//
func (p *gcParser) parseField() (*Var, string) {
	pkg, name := p.parseName(true)
	typ := p.parseType()
	anonymous := false
	if name == "" {
		// anonymous field - typ must be T or *T and T must be a type name
		switch typ, _ := deref(typ); typ := typ.(type) {
		case *Basic: // basic types are named types
			pkg = nil
			name = typ.name
		case *Named:
			pkg = typ.obj.pkg
			name = typ.obj.name
		default:
			p.errorf("anonymous field expected")
		}
		anonymous = true
	}
	tag := ""
	if p.tok == scanner.String {
		tag = p.expect(scanner.String)
	}
	return NewField(token.NoPos, pkg, name, typ, anonymous), tag
}

// StructType = "struct" "{" [ FieldList ] "}" .
// FieldList  = Field { ";" Field } .
//
func (p *gcParser) parseStructType() Type {
	var fields []*Var
	var tags []string

	p.expectKeyword("struct")
	p.expect('{')
	var fset objset
	for i := 0; p.tok != '}'; i++ {
		if i > 0 {
			p.expect(';')
		}
		fld, tag := p.parseField()
		if tag != "" && tags == nil {
			tags = make([]string, i)
		}
		if tags != nil {
			tags = append(tags, tag)
		}
		if alt := fset.insert(fld); alt != nil {
			pname := "<no pkg name>"
			if pkg := alt.Pkg(); pkg != nil {
				pname = pkg.name
			}
			p.errorf("multiple fields named %s.%s", pname, alt.Name())
			continue
		}
		fields = append(fields, fld)
	}
	p.expect('}')

	return &Struct{fields: fields, tags: tags}
}

// Parameter = ( identifier | "?" ) [ "..." ] Type [ string_lit ] .
//
func (p *gcParser) parseParameter() (par *Var, isVariadic bool) {
	_, name := p.parseName(false)
	if name == "" {
		name = "_" // cannot access unnamed identifiers
	}
	if p.tok == '.' {
		p.expectSpecial("...")
		isVariadic = true
	}
	typ := p.parseType()
	if isVariadic {
		typ = &Slice{elt: typ}
	}
	// ignore argument tag (e.g. "noescape")
	if p.tok == scanner.String {
		p.next()
	}
	// TODO(gri) should we provide a package?
	par = NewVar(token.NoPos, nil, name, typ)
	return
}

// Parameters    = "(" [ ParameterList ] ")" .
// ParameterList = { Parameter "," } Parameter .
//
func (p *gcParser) parseParameters() (list []*Var, isVariadic bool) {
	p.expect('(')
	for p.tok != ')' {
		if len(list) > 0 {
			p.expect(',')
		}
		par, variadic := p.parseParameter()
		list = append(list, par)
		if variadic {
			if isVariadic {
				p.error("... not on final argument")
			}
			isVariadic = true
		}
	}
	p.expect(')')

	return
}

// Signature = Parameters [ Result ] .
// Result    = Type | Parameters .
//
func (p *gcParser) parseSignature() *Signature {
	params, isVariadic := p.parseParameters()

	// optional result type
	var results []*Var
	if p.tok == '(' {
		var variadic bool
		results, variadic = p.parseParameters()
		if variadic {
			p.error("... not permitted on result type")
		}
	}

	return &Signature{params: NewTuple(params...), results: NewTuple(results...), isVariadic: isVariadic}
}

// InterfaceType = "interface" "{" [ MethodList ] "}" .
// MethodList    = Method { ";" Method } .
// Method        = Name Signature .
//
// The methods of embedded interfaces are always "inlined"
// by the compiler and thus embedded interfaces are never
// visible in the export data.
//
func (p *gcParser) parseInterfaceType() Type {
	typ := new(Interface)
	var methods []*Func

	p.expectKeyword("interface")
	p.expect('{')
	var mset objset
	for i := 0; p.tok != '}'; i++ {
		if i > 0 {
			p.expect(';')
		}
		pkg, name := p.parseName(true)
		sig := p.parseSignature()
		// TODO(gri) Ideally, we should use a named type here instead of
		// typ, for less verbose printing of interface method signatures.
		sig.recv = NewVar(token.NoPos, pkg, "", typ)
		m := NewFunc(token.NoPos, pkg, name, sig)
		if alt := mset.insert(m); alt != nil {
			p.errorf("multiple methods named %s.%s", alt.Pkg().name, alt.Name())
			continue
		}
		methods = append(methods, m)
	}
	p.expect('}')

	typ.methods = methods
	return typ
}

// ChanType = ( "chan" [ "<-" ] | "<-" "chan" ) Type .
//
func (p *gcParser) parseChanType() Type {
	dir := ast.SEND | ast.RECV
	if p.tok == scanner.Ident {
		p.expectKeyword("chan")
		if p.tok == '<' {
			p.expectSpecial("<-")
			dir = ast.SEND
		}
	} else {
		p.expectSpecial("<-")
		p.expectKeyword("chan")
		dir = ast.RECV
	}
	elt := p.parseType()
	return &Chan{dir: dir, elt: elt}
}

// Type =
//	BasicType | TypeName | ArrayType | SliceType | StructType |
//      PointerType | FuncType | InterfaceType | MapType | ChanType |
//      "(" Type ")" .
//
// BasicType   = ident .
// TypeName    = ExportedName .
// SliceType   = "[" "]" Type .
// PointerType = "*" Type .
// FuncType    = "func" Signature .
//
func (p *gcParser) parseType() Type {
	switch p.tok {
	case scanner.Ident:
		switch p.lit {
		default:
			return p.parseBasicType()
		case "struct":
			return p.parseStructType()
		case "func":
			// FuncType
			p.next()
			return p.parseSignature()
		case "interface":
			return p.parseInterfaceType()
		case "map":
			return p.parseMapType()
		case "chan":
			return p.parseChanType()
		}
	case '@':
		// TypeName
		pkg, name := p.parseExportedName()
		return declTypeName(pkg, name).typ
	case '[':
		p.next() // look ahead
		if p.tok == ']' {
			// SliceType
			p.next()
			return &Slice{elt: p.parseType()}
		}
		return p.parseArrayType()
	case '*':
		// PointerType
		p.next()
		return &Pointer{base: p.parseType()}
	case '<':
		return p.parseChanType()
	case '(':
		// "(" Type ")"
		p.next()
		typ := p.parseType()
		p.expect(')')
		return typ
	}
	p.errorf("expected type, got %s (%q)", scanner.TokenString(p.tok), p.lit)
	return nil
}

// ----------------------------------------------------------------------------
// Declarations

// ImportDecl = "import" PackageName PackageId .
//
func (p *gcParser) parseImportDecl() {
	p.expectKeyword("import")
	name := p.parsePackageName()
	p.getPkg(p.parsePackageId(), name)
}

// int_lit = [ "+" | "-" ] { "0" ... "9" } .
//
func (p *gcParser) parseInt() string {
	s := ""
	switch p.tok {
	case '-':
		s = "-"
		p.next()
	case '+':
		p.next()
	}
	return s + p.expect(scanner.Int)
}

// number = int_lit [ "p" int_lit ] .
//
func (p *gcParser) parseNumber() (x operand) {
	x.mode = constant

	// mantissa
	mant := exact.MakeFromLiteral(p.parseInt(), token.INT)
	assert(mant != nil)

	if p.lit == "p" {
		// exponent (base 2)
		p.next()
		exp, err := strconv.ParseInt(p.parseInt(), 10, 0)
		if err != nil {
			p.error(err)
		}
		if exp < 0 {
			denom := exact.MakeInt64(1)
			denom = exact.Shift(denom, token.SHL, uint(-exp))
			x.typ = Typ[UntypedFloat]
			x.val = exact.BinaryOp(mant, token.QUO, denom)
			return
		}
		if exp > 0 {
			mant = exact.Shift(mant, token.SHL, uint(exp))
		}
		x.typ = Typ[UntypedFloat]
		x.val = mant
		return
	}

	x.typ = Typ[UntypedInt]
	x.val = mant
	return
}

// ConstDecl   = "const" ExportedName [ Type ] "=" Literal .
// Literal     = bool_lit | int_lit | float_lit | complex_lit | rune_lit | string_lit .
// bool_lit    = "true" | "false" .
// complex_lit = "(" float_lit "+" float_lit "i" ")" .
// rune_lit    = "(" int_lit "+" int_lit ")" .
// string_lit  = `"` { unicode_char } `"` .
//
func (p *gcParser) parseConstDecl() {
	p.expectKeyword("const")
	pkg, name := p.parseExportedName()
	obj := declConst(pkg, name)
	var x operand
	if p.tok != '=' {
		obj.typ = p.parseType()
	}
	p.expect('=')
	switch p.tok {
	case scanner.Ident:
		// bool_lit
		if p.lit != "true" && p.lit != "false" {
			p.error("expected true or false")
		}
		x.typ = Typ[UntypedBool]
		x.val = exact.MakeBool(p.lit == "true")
		p.next()

	case '-', scanner.Int:
		// int_lit
		x = p.parseNumber()

	case '(':
		// complex_lit or rune_lit
		p.next()
		if p.tok == scanner.Char {
			p.next()
			p.expect('+')
			x = p.parseNumber()
			x.typ = Typ[UntypedRune]
			p.expect(')')
			break
		}
		re := p.parseNumber()
		p.expect('+')
		im := p.parseNumber()
		p.expectKeyword("i")
		p.expect(')')
		x.typ = Typ[UntypedComplex]
		// TODO(gri) fix this
		_, _ = re, im
		x.val = exact.MakeInt64(0)

	case scanner.Char:
		// rune_lit
		x.setConst(token.CHAR, p.lit)
		p.next()

	case scanner.String:
		// string_lit
		x.setConst(token.STRING, p.lit)
		p.next()

	default:
		p.errorf("expected literal got %s", scanner.TokenString(p.tok))
	}
	if obj.typ == nil {
		obj.typ = x.typ
	}
	assert(x.val != nil)
	obj.val = x.val
}

// TypeDecl = "type" ExportedName Type .
//
func (p *gcParser) parseTypeDecl() {
	p.expectKeyword("type")
	pkg, name := p.parseExportedName()
	obj := declTypeName(pkg, name)

	// The type object may have been imported before and thus already
	// have a type associated with it. We still need to parse the type
	// structure, but throw it away if the object already has a type.
	// This ensures that all imports refer to the same type object for
	// a given type declaration.
	typ := p.parseType()

	if name := obj.typ.(*Named); name.underlying == nil {
		name.underlying = typ
		name.complete = true
	}
}

// VarDecl = "var" ExportedName Type .
//
func (p *gcParser) parseVarDecl() {
	p.expectKeyword("var")
	pkg, name := p.parseExportedName()
	obj := declVar(pkg, name)
	obj.typ = p.parseType()
}

// Func = Signature [ Body ] .
// Body = "{" ... "}" .
//
func (p *gcParser) parseFunc() *Signature {
	sig := p.parseSignature()
	if p.tok == '{' {
		p.next()
		for i := 1; i > 0; p.next() {
			switch p.tok {
			case '{':
				i++
			case '}':
				i--
			}
		}
	}
	return sig
}

// MethodDecl = "func" Receiver Name Func .
// Receiver   = "(" ( identifier | "?" ) [ "*" ] ExportedName ")" .
//
func (p *gcParser) parseMethodDecl() {
	// "func" already consumed
	p.expect('(')
	recv, _ := p.parseParameter() // receiver
	p.expect(')')

	// determine receiver base type object
	typ := recv.typ
	if ptr, ok := typ.(*Pointer); ok {
		typ = ptr.base
	}
	base := typ.(*Named)

	// parse method name, signature, and possibly inlined body
	pkg, name := p.parseName(true)
	sig := p.parseFunc()
	sig.recv = recv

	// add method to type unless type was imported before
	// and method exists already
	// TODO(gri) This is a quadratic algorithm - ok for now because method counts are small.
	if _, m := lookupMethod(base.methods, pkg, name); m == nil {
		base.methods = append(base.methods, NewFunc(token.NoPos, pkg, name, sig))
	}
}

// FuncDecl = "func" ExportedName Func .
//
func (p *gcParser) parseFuncDecl() {
	// "func" already consumed
	pkg, name := p.parseExportedName()
	typ := p.parseFunc()
	declFunc(pkg, name).typ = typ
}

// Decl = [ ImportDecl | ConstDecl | TypeDecl | VarDecl | FuncDecl | MethodDecl ] "\n" .
//
func (p *gcParser) parseDecl() {
	switch p.lit {
	case "import":
		p.parseImportDecl()
	case "const":
		p.parseConstDecl()
	case "type":
		p.parseTypeDecl()
	case "var":
		p.parseVarDecl()
	case "func":
		p.next() // look ahead
		if p.tok == '(' {
			p.parseMethodDecl()
		} else {
			p.parseFuncDecl()
		}
	}
	p.expect('\n')
}

// ----------------------------------------------------------------------------
// Export

// Export        = "PackageClause { Decl } "$$" .
// PackageClause = "package" PackageName [ "safe" ] "\n" .
//
func (p *gcParser) parseExport() *Package {
	p.expectKeyword("package")
	name := p.parsePackageName()
	if p.tok != '\n' {
		// A package is safe if it was compiled with the -u flag,
		// which disables the unsafe package.
		// TODO(gri) remember "safe" package
		p.expectKeyword("safe")
	}
	p.expect('\n')

	pkg := p.getPkg(p.id, name)

	for p.tok != '$' && p.tok != scanner.EOF {
		p.parseDecl()
	}

	if ch := p.scanner.Peek(); p.tok != '$' || ch != '$' {
		// don't call next()/expect() since reading past the
		// export data may cause scanner errors (e.g. NUL chars)
		p.errorf("expected '$$', got %s %c", scanner.TokenString(p.tok), ch)
	}

	if n := p.scanner.ErrorCount; n != 0 {
		p.errorf("expected no scanner errors, got %d", n)
	}

	// package was imported completely and without errors
	pkg.complete = true

	return pkg
}

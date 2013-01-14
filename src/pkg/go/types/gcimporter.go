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
	"math/big"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"text/scanner"
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
		noext = bp.PkgObj
		if strings.HasSuffix(noext, ".a") {
			noext = noext[:len(noext)-len(".a")]
		}

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
		if r := recover(); r != nil {
			err = r.(importError) // will re-panic if r is not an importError
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
// GcImport satisfies the ast.Importer signature.
//
func GcImport(imports map[string]*Package, path string) (pkg *Package, err error) {
	if path == "unsafe" {
		return Unsafe, nil
	}

	srcDir, err := os.Getwd()
	if err != nil {
		return
	}
	filename, id := FindPkg(path, srcDir)
	if filename == "" {
		err = errors.New("can't find import: " + id)
		return
	}

	// no need to re-import if the package was imported completely before
	if pkg = imports[id]; pkg != nil && pkg.Complete {
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
			// Add file name to error.
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

func declConst(scope *Scope, name string) *Const {
	// the constant may have been imported before - if it exists
	// already in the respective scope, return that constant
	if obj := scope.Lookup(name); obj != nil {
		return obj.(*Const)
	}
	// otherwise create a new constant and insert it into the scope
	obj := &Const{Name: name}
	scope.Insert(obj)
	return obj
}

func declTypeName(scope *Scope, name string) *TypeName {
	if obj := scope.Lookup(name); obj != nil {
		return obj.(*TypeName)
	}
	obj := &TypeName{Name: name}
	// a named type may be referred to before the underlying type
	// is known - set it up
	obj.Type = &NamedType{Obj: obj}
	scope.Insert(obj)
	return obj
}

func declVar(scope *Scope, name string) *Var {
	if obj := scope.Lookup(name); obj != nil {
		return obj.(*Var)
	}
	obj := &Var{Name: name}
	scope.Insert(obj)
	return obj
}

func declFunc(scope *Scope, name string) *Func {
	if obj := scope.Lookup(name); obj != nil {
		return obj.(*Func)
	}
	obj := &Func{Name: name}
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
// Import declarations

// ImportPath = string_lit .
//
func (p *gcParser) parsePkgId() *Package {
	id, err := strconv.Unquote(p.expect(scanner.String))
	if err != nil {
		p.error(err)
	}

	switch id {
	case "":
		// id == "" stands for the imported package id
		// (only known at time of package installation)
		id = p.id
	case "unsafe":
		// package unsafe is not in the imports map - handle explicitly
		return Unsafe
	}

	pkg := p.imports[id]
	if pkg == nil {
		pkg = &Package{Scope: new(Scope)}
		p.imports[id] = pkg
	}

	return pkg
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

// ExportedName = "@" ImportPath "." dotIdentifier .
//
func (p *gcParser) parseExportedName() (*Package, string) {
	p.expect('@')
	pkg := p.parsePkgId()
	p.expect('.')
	name := p.parseDotIdent()
	return pkg, name
}

// ----------------------------------------------------------------------------
// Types

// BasicType = identifier .
//
func (p *gcParser) parseBasicType() Type {
	id := p.expect(scanner.Ident)
	obj := Universe.Lookup(id)
	if obj, ok := obj.(*TypeName); ok {
		return obj.Type
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
	return &Array{Len: n, Elt: elt}
}

// MapType = "map" "[" Type "]" Type .
//
func (p *gcParser) parseMapType() Type {
	p.expectKeyword("map")
	p.expect('[')
	key := p.parseType()
	p.expect(']')
	elt := p.parseType()
	return &Map{Key: key, Elt: elt}
}

// Name = identifier | "?" | ExportedName  .
//
func (p *gcParser) parseName() (pkg *Package, name string) {
	switch p.tok {
	case scanner.Ident:
		name = p.lit
		p.next()
	case '?':
		// anonymous
		p.next()
	case '@':
		// exported name prefixed with package path
		pkg, name = p.parseExportedName()
	default:
		p.error("name expected")
	}
	return
}

// Field = Name Type [ string_lit ] .
//
func (p *gcParser) parseField() *Field {
	var f Field
	f.Pkg, f.Name = p.parseName()
	f.Type = p.parseType()
	if p.tok == scanner.String {
		f.Tag = p.expect(scanner.String)
	}
	if f.Name == "" {
		// anonymous field - typ must be T or *T and T must be a type name
		if typ, ok := deref(f.Type).(*NamedType); ok && typ.Obj != nil {
			f.Name = typ.Obj.GetName()
			f.IsAnonymous = true
		} else {
			p.errorf("anonymous field expected")
		}
	}
	return &f
}

// StructType = "struct" "{" [ FieldList ] "}" .
// FieldList  = Field { ";" Field } .
//
func (p *gcParser) parseStructType() Type {
	var fields []*Field

	p.expectKeyword("struct")
	p.expect('{')
	for p.tok != '}' {
		if len(fields) > 0 {
			p.expect(';')
		}
		fields = append(fields, p.parseField())
	}
	p.expect('}')

	return &Struct{Fields: fields}
}

// Parameter = ( identifier | "?" ) [ "..." ] Type [ string_lit ] .
//
func (p *gcParser) parseParameter() (par *Var, isVariadic bool) {
	_, name := p.parseName()
	if name == "" {
		name = "_" // cannot access unnamed identifiers
	}
	if p.tok == '.' {
		p.expectSpecial("...")
		isVariadic = true
	}
	typ := p.parseType()
	// ignore argument tag (e.g. "noescape")
	if p.tok == scanner.String {
		p.next()
	}
	par = &Var{Name: name, Type: typ}
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
	switch p.tok {
	case scanner.Ident, '[', '*', '<', '@':
		// single, unnamed result
		results = []*Var{{Type: p.parseType()}}
	case '(':
		// named or multiple result(s)
		var variadic bool
		results, variadic = p.parseParameters()
		if variadic {
			p.error("... not permitted on result type")
		}
	}

	return &Signature{Params: params, Results: results, IsVariadic: isVariadic}
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
	var methods []*Method

	p.expectKeyword("interface")
	p.expect('{')
	for p.tok != '}' {
		if len(methods) > 0 {
			p.expect(';')
		}
		pkg, name := p.parseName()
		typ := p.parseSignature()
		methods = append(methods, &Method{QualifiedName{pkg, name}, typ})
	}
	p.expect('}')

	return &Interface{Methods: methods}
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
	return &Chan{Dir: dir, Elt: elt}
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
		return declTypeName(pkg.Scope, name).Type
	case '[':
		p.next() // look ahead
		if p.tok == ']' {
			// SliceType
			p.next()
			return &Slice{Elt: p.parseType()}
		}
		return p.parseArrayType()
	case '*':
		// PointerType
		p.next()
		return &Pointer{Base: p.parseType()}
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

// ImportDecl = "import" identifier string_lit .
//
func (p *gcParser) parseImportDecl() {
	p.expectKeyword("import")
	// The identifier has no semantic meaning in the import data.
	// It exists so that error messages can print the real package
	// name: binary.ByteOrder instead of "encoding/binary".ByteOrder.
	name := p.expect(scanner.Ident)
	pkg := p.parsePkgId()
	assert(pkg.Name == "" || pkg.Name == name)
	pkg.Name = name
}

// int_lit = [ "+" | "-" ] { "0" ... "9" } .
//
func (p *gcParser) parseInt() (neg bool, val string) {
	switch p.tok {
	case '-':
		neg = true
		fallthrough
	case '+':
		p.next()
	}
	val = p.expect(scanner.Int)
	return
}

// number = int_lit [ "p" int_lit ] .
//
func (p *gcParser) parseNumber() (x operand) {
	x.mode = constant

	// mantissa
	neg, val := p.parseInt()
	mant, ok := new(big.Int).SetString(val, 0)
	assert(ok)
	if neg {
		mant.Neg(mant)
	}

	if p.lit == "p" {
		// exponent (base 2)
		p.next()
		neg, val = p.parseInt()
		exp64, err := strconv.ParseUint(val, 10, 0)
		if err != nil {
			p.error(err)
		}
		exp := uint(exp64)
		if neg {
			denom := big.NewInt(1)
			denom.Lsh(denom, exp)
			x.typ = Typ[UntypedFloat]
			x.val = normalizeRatConst(new(big.Rat).SetFrac(mant, denom))
			return
		}
		if exp > 0 {
			mant.Lsh(mant, exp)
		}
		x.typ = Typ[UntypedFloat]
		x.val = normalizeIntConst(mant)
		return
	}

	x.typ = Typ[UntypedInt]
	x.val = normalizeIntConst(mant)
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
	obj := declConst(pkg.Scope, name)
	var x operand
	if p.tok != '=' {
		obj.Type = p.parseType()
	}
	p.expect('=')
	switch p.tok {
	case scanner.Ident:
		// bool_lit
		if p.lit != "true" && p.lit != "false" {
			p.error("expected true or false")
		}
		x.typ = Typ[UntypedBool]
		x.val = p.lit == "true"
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
		x.val = zeroConst

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
	if obj.Type == nil {
		obj.Type = x.typ
	}
	assert(x.val != nil)
	obj.Val = x.val
}

// TypeDecl = "type" ExportedName Type .
//
func (p *gcParser) parseTypeDecl() {
	p.expectKeyword("type")
	pkg, name := p.parseExportedName()
	obj := declTypeName(pkg.Scope, name)

	// The type object may have been imported before and thus already
	// have a type associated with it. We still need to parse the type
	// structure, but throw it away if the object already has a type.
	// This ensures that all imports refer to the same type object for
	// a given type declaration.
	typ := p.parseType()

	if name := obj.Type.(*NamedType); name.Underlying == nil {
		name.Underlying = typ
	}
}

// VarDecl = "var" ExportedName Type .
//
func (p *gcParser) parseVarDecl() {
	p.expectKeyword("var")
	pkg, name := p.parseExportedName()
	obj := declVar(pkg.Scope, name)
	obj.Type = p.parseType()
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
	typ := recv.Type
	if ptr, ok := typ.(*Pointer); ok {
		typ = ptr.Base
	}
	base := typ.(*NamedType)

	// parse method name, signature, and possibly inlined body
	pkg, name := p.parseName() // unexported method names in imports are qualified with their package.
	sig := p.parseFunc()
	sig.Recv = recv

	// add method to type unless type was imported before
	// and method exists already
	// TODO(gri) investigate if this can be avoided
	for _, m := range base.Methods {
		if m.Name == name {
			return // method was added before
		}
	}
	base.Methods = append(base.Methods, &Method{QualifiedName{pkg, name}, sig})
}

// FuncDecl = "func" ExportedName Func .
//
func (p *gcParser) parseFuncDecl() {
	// "func" already consumed
	pkg, name := p.parseExportedName()
	typ := p.parseFunc()
	declFunc(pkg.Scope, name).Type = typ
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
// PackageClause = "package" identifier [ "safe" ] "\n" .
//
func (p *gcParser) parseExport() *Package {
	p.expectKeyword("package")
	name := p.expect(scanner.Ident)
	if p.tok != '\n' {
		// A package is safe if it was compiled with the -u flag,
		// which disables the unsafe package.
		// TODO(gri) remember "safe" package
		p.expectKeyword("safe")
	}
	p.expect('\n')

	pkg := p.imports[p.id]
	if pkg == nil {
		pkg = &Package{Name: name, Scope: new(Scope)}
		p.imports[p.id] = pkg
	}

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
	pkg.Complete = true

	return pkg
}

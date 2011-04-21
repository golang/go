// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements an ast.Importer for gc generated object files.
// TODO(gri) Eventually move this into a separate package outside types.

package types

import (
	"big"
	"fmt"
	"go/ast"
	"go/token"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"scanner"
	"strconv"
)


const trace = false // set to true for debugging

var (
	pkgRoot = filepath.Join(runtime.GOROOT(), "pkg", runtime.GOOS+"_"+runtime.GOARCH)
	pkgExts = [...]string{".a", ".5", ".6", ".8"}
)


// findPkg returns the filename and package id for an import path.
// If no file was found, an empty filename is returned.
func findPkg(path string) (filename, id string) {
	if len(path) == 0 {
		return
	}

	id = path
	var noext string
	switch path[0] {
	default:
		// "x" -> "$GOROOT/pkg/$GOOS_$GOARCH/x.ext", "x"
		noext = filepath.Join(pkgRoot, path)

	case '.':
		// "./x" -> "/this/directory/x.ext", "/this/directory/x"
		cwd, err := os.Getwd()
		if err != nil {
			return
		}
		noext = filepath.Join(cwd, path)
		id = noext

	case '/':
		// "/x" -> "/x.ext", "/x"
		noext = path
	}

	// try extensions
	for _, ext := range pkgExts {
		filename = noext + ext
		if f, err := os.Stat(filename); err == nil && f.IsRegular() {
			return
		}
	}

	filename = "" // not found
	return
}


// gcParser parses the exports inside a gc compiler-produced
// object/archive file and populates its scope with the results.
type gcParser struct {
	scanner scanner.Scanner
	tok     int                   // current token
	lit     string                // literal string; only valid for Ident, Int, String tokens
	id      string                // package id of imported package
	scope   *ast.Scope            // scope of imported package; alias for deps[id]
	deps    map[string]*ast.Scope // package id -> package scope
}


func (p *gcParser) init(filename, id string, src io.Reader) {
	p.scanner.Init(src)
	p.scanner.Error = func(_ *scanner.Scanner, msg string) { p.error(msg) }
	p.scanner.Mode = scanner.ScanIdents | scanner.ScanInts | scanner.ScanStrings | scanner.ScanComments | scanner.SkipComments
	p.scanner.Whitespace = 1<<'\t' | 1<<' '
	p.scanner.Filename = filename // for good error messages
	p.next()
	p.id = id
	p.scope = ast.NewScope(nil)
	p.deps = map[string]*ast.Scope{"unsafe": Unsafe, id: p.scope}
}


func (p *gcParser) next() {
	p.tok = p.scanner.Scan()
	switch p.tok {
	case scanner.Ident, scanner.Int, scanner.String:
		p.lit = p.scanner.TokenText()
	default:
		p.lit = ""
	}
	if trace {
		fmt.Printf("%s: %q -> %q\n", scanner.TokenString(p.tok), p.scanner.TokenText(), p.lit)
	}
}


// GcImporter implements the ast.Importer signature.
func GcImporter(path string) (name string, scope *ast.Scope, err os.Error) {
	if path == "unsafe" {
		return path, Unsafe, nil
	}

	defer func() {
		if r := recover(); r != nil {
			err = r.(importError) // will re-panic if r is not an importError
			if trace {
				panic(err) // force a stack trace
			}
		}
	}()

	filename, id := findPkg(path)
	if filename == "" {
		err = os.ErrorString("can't find import: " + id)
		return
	}

	buf, err := ExportData(filename)
	if err != nil {
		return
	}
	defer buf.Close()

	if trace {
		fmt.Printf("importing %s\n", filename)
	}

	var p gcParser
	p.init(filename, id, buf)
	name, scope = p.parseExport()

	return
}


// ----------------------------------------------------------------------------
// Error handling

// Internal errors are boxed as importErrors.
type importError struct {
	pos scanner.Position
	err os.Error
}


func (e importError) String() string {
	return fmt.Sprintf("import error %s (byte offset = %d): %s", e.pos, e.pos.Offset, e.err)
}


func (p *gcParser) error(err interface{}) {
	if s, ok := err.(string); ok {
		err = os.ErrorString(s)
	}
	// panic with a runtime.Error if err is not an os.Error
	panic(importError{p.scanner.Pos(), err.(os.Error)})
}


func (p *gcParser) errorf(format string, args ...interface{}) {
	p.error(fmt.Sprintf(format, args...))
}


func (p *gcParser) expect(tok int) string {
	lit := p.lit
	if p.tok != tok {
		p.errorf("expected %q, got %q (%q)", scanner.TokenString(tok), scanner.TokenString(p.tok), lit)
	}
	p.next()
	return lit
}


func (p *gcParser) expectSpecial(tok string) {
	sep := 'x' // not white space
	i := 0
	for i < len(tok) && p.tok == int(tok[i]) && sep > ' ' {
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
func (p *gcParser) parsePkgId() *ast.Scope {
	id, err := strconv.Unquote(p.expect(scanner.String))
	if err != nil {
		p.error(err)
	}

	scope := p.scope // id == "" stands for the imported package id
	if id != "" {
		if scope = p.deps[id]; scope == nil {
			scope = ast.NewScope(nil)
			p.deps[id] = scope
		}
	}

	return scope
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


// ExportedName = ImportPath "." dotIdentifier .
//
func (p *gcParser) parseExportedName(kind ast.ObjKind) *ast.Object {
	scope := p.parsePkgId()
	p.expect('.')
	name := p.parseDotIdent()

	// a type may have been declared before - if it exists
	// already in the respective package scope, return that
	// type
	if kind == ast.Typ {
		if obj := scope.Lookup(name); obj != nil {
			assert(obj.Kind == ast.Typ)
			return obj
		}
	}

	// any other object must be a newly declared object -
	// create it and insert it into the package scope
	obj := ast.NewObj(kind, name)
	if scope.Insert(obj) != nil {
		p.errorf("already declared: %s", obj.Name)
	}

	// a new type object is a named type and may be referred
	// to before the underlying type is known - set it up
	if kind == ast.Typ {
		obj.Type = &Name{Obj: obj}
	}

	return obj
}


// ----------------------------------------------------------------------------
// Types

// BasicType = identifier .
//
func (p *gcParser) parseBasicType() Type {
	obj := Universe.Lookup(p.expect(scanner.Ident))
	if obj == nil || obj.Kind != ast.Typ {
		p.errorf("not a basic type: %s", obj.Name)
	}
	return obj.Type.(Type)
}


// ArrayType = "[" int_lit "]" Type .
//
func (p *gcParser) parseArrayType() Type {
	// "[" already consumed and lookahead known not to be "]"
	lit := p.expect(scanner.Int)
	p.expect(']')
	elt := p.parseType()
	n, err := strconv.Atoui64(lit)
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


// Name = identifier | "?" .
//
func (p *gcParser) parseName() (name string) {
	switch p.tok {
	case scanner.Ident:
		name = p.lit
		p.next()
	case '?':
		// anonymous
		p.next()
	default:
		p.error("name expected")
	}
	return
}


// Field = Name Type [ ":" string_lit ] .
//
func (p *gcParser) parseField(scope *ast.Scope) {
	// TODO(gri) The code below is not correct for anonymous fields:
	//           The name is the type name; it should not be empty.
	name := p.parseName()
	ftyp := p.parseType()
	if name == "" {
		// anonymous field - ftyp must be T or *T and T must be a type name
		ftyp = Deref(ftyp)
		if ftyp, ok := ftyp.(*Name); ok {
			name = ftyp.Obj.Name
		} else {
			p.errorf("anonymous field expected")
		}
	}
	if p.tok == ':' {
		p.next()
		tag := p.expect(scanner.String)
		_ = tag // TODO(gri) store tag somewhere
	}
	fld := ast.NewObj(ast.Var, name)
	fld.Type = ftyp
	scope.Insert(fld)
}


// StructType = "struct" "{" [ FieldList ] "}" .
// FieldList  = Field { ";" Field } .
//
func (p *gcParser) parseStructType() Type {
	p.expectKeyword("struct")
	p.expect('{')
	scope := ast.NewScope(nil)
	if p.tok != '}' {
		p.parseField(scope)
		for p.tok == ';' {
			p.next()
			p.parseField(scope)
		}
	}
	p.expect('}')
	return &Struct{}
}


// Parameter = ( identifier | "?" ) [ "..." ] Type .
//
func (p *gcParser) parseParameter(scope *ast.Scope, isVariadic *bool) {
	name := p.parseName()
	if name == "" {
		name = "_" // cannot access unnamed identifiers
	}
	if isVariadic != nil {
		if *isVariadic {
			p.error("... not on final argument")
		}
		if p.tok == '.' {
			p.expectSpecial("...")
			*isVariadic = true
		}
	}
	ptyp := p.parseType()
	par := ast.NewObj(ast.Var, name)
	par.Type = ptyp
	scope.Insert(par)
}


// Parameters    = "(" [ ParameterList ] ")" .
// ParameterList = { Parameter "," } Parameter .
//
func (p *gcParser) parseParameters(scope *ast.Scope, isVariadic *bool) {
	p.expect('(')
	if p.tok != ')' {
		p.parseParameter(scope, isVariadic)
		for p.tok == ',' {
			p.next()
			p.parseParameter(scope, isVariadic)
		}
	}
	p.expect(')')
}


// Signature = Parameters [ Result ] .
// Result    = Type | Parameters .
//
func (p *gcParser) parseSignature(scope *ast.Scope, isVariadic *bool) {
	p.parseParameters(scope, isVariadic)

	// optional result type
	switch p.tok {
	case scanner.Ident, scanner.String, '[', '*', '<':
		// single, unnamed result
		result := ast.NewObj(ast.Var, "_")
		result.Type = p.parseType()
		scope.Insert(result)
	case '(':
		// named or multiple result(s)
		p.parseParameters(scope, nil)
	}
}


// FuncType = "func" Signature .
//
func (p *gcParser) parseFuncType() Type {
	// "func" already consumed
	scope := ast.NewScope(nil)
	isVariadic := false
	p.parseSignature(scope, &isVariadic)
	return &Func{IsVariadic: isVariadic}
}


// MethodSpec = identifier Signature .
//
func (p *gcParser) parseMethodSpec(scope *ast.Scope) {
	if p.tok == scanner.Ident {
		p.expect(scanner.Ident)
	} else {
		p.parsePkgId()
		p.expect('.')
		p.parseDotIdent()
	}
	isVariadic := false
	p.parseSignature(scope, &isVariadic)
}


// InterfaceType = "interface" "{" [ MethodList ] "}" .
// MethodList    = MethodSpec { ";" MethodSpec } .
//
func (p *gcParser) parseInterfaceType() Type {
	p.expectKeyword("interface")
	p.expect('{')
	scope := ast.NewScope(nil)
	if p.tok != '}' {
		p.parseMethodSpec(scope)
		for p.tok == ';' {
			p.next()
			p.parseMethodSpec(scope)
		}
	}
	p.expect('}')
	return &Interface{}
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
// BasicType = ident .
// TypeName = ExportedName .
// SliceType = "[" "]" Type .
// PointerType = "*" Type .
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
			p.next() // parseFuncType assumes "func" is already consumed
			return p.parseFuncType()
		case "interface":
			return p.parseInterfaceType()
		case "map":
			return p.parseMapType()
		case "chan":
			return p.parseChanType()
		}
	case scanner.String:
		// TypeName
		return p.parseExportedName(ast.Typ).Type.(Type)
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
	// TODO(gri): Save package id -> package name mapping.
	p.expect(scanner.Ident)
	p.parsePkgId()
}


// int_lit = [ "+" | "-" ] { "0" ... "9" } .
//
func (p *gcParser) parseInt() (sign, val string) {
	switch p.tok {
	case '-':
		p.next()
		sign = "-"
	case '+':
		p.next()
	}
	val = p.expect(scanner.Int)
	return
}


// number = int_lit [ "p" int_lit ] .
//
func (p *gcParser) parseNumber() Const {
	// mantissa
	sign, val := p.parseInt()
	mant, ok := new(big.Int).SetString(sign+val, 10)
	assert(ok)

	if p.lit == "p" {
		// exponent (base 2)
		p.next()
		sign, val = p.parseInt()
		exp, err := strconv.Atoui(val)
		if err != nil {
			p.error(err)
		}
		if sign == "-" {
			denom := big.NewInt(1)
			denom.Lsh(denom, exp)
			return Const{new(big.Rat).SetFrac(mant, denom)}
		}
		if exp > 0 {
			mant.Lsh(mant, exp)
		}
		return Const{new(big.Rat).SetInt(mant)}
	}

	return Const{mant}
}


// ConstDecl   = "const" ExportedName [ Type ] "=" Literal .
// Literal     = bool_lit | int_lit | float_lit | complex_lit | string_lit .
// bool_lit    = "true" | "false" .
// complex_lit = "(" float_lit "+" float_lit ")" .
// string_lit  = `"` { unicode_char } `"` .
//
func (p *gcParser) parseConstDecl() {
	p.expectKeyword("const")
	obj := p.parseExportedName(ast.Con)
	var x Const
	var typ Type
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
		x = Const{p.lit == "true"}
		typ = Bool.Underlying
		p.next()
	case '-', scanner.Int:
		// int_lit
		x = p.parseNumber()
		typ = Int.Underlying
		if _, ok := x.val.(*big.Rat); ok {
			typ = Float64.Underlying
		}
	case '(':
		// complex_lit
		p.next()
		re := p.parseNumber()
		p.expect('+')
		im := p.parseNumber()
		p.expect(')')
		x = Const{cmplx{re.val.(*big.Rat), im.val.(*big.Rat)}}
		typ = Complex128.Underlying
	case scanner.String:
		// string_lit
		x = MakeConst(token.STRING, p.lit)
		p.next()
		typ = String.Underlying
	default:
		p.error("expected literal")
	}
	if obj.Type == nil {
		obj.Type = typ
	}
	_ = x // TODO(gri) store x somewhere
}


// TypeDecl = "type" ExportedName Type .
//
func (p *gcParser) parseTypeDecl() {
	p.expectKeyword("type")
	obj := p.parseExportedName(ast.Typ)
	typ := p.parseType()

	name := obj.Type.(*Name)
	assert(name.Underlying == nil)
	assert(Underlying(typ) == typ)
	name.Underlying = typ
}


// VarDecl = "var" ExportedName Type .
//
func (p *gcParser) parseVarDecl() {
	p.expectKeyword("var")
	obj := p.parseExportedName(ast.Var)
	obj.Type = p.parseType()
}


// FuncDecl = "func" ExportedName Signature .
//
func (p *gcParser) parseFuncDecl() {
	// "func" already consumed
	obj := p.parseExportedName(ast.Fun)
	obj.Type = p.parseFuncType()
}


// MethodDecl = "func" Receiver identifier Signature .
// Receiver   = "(" ( identifier | "?" ) [ "*" ] ExportedName ")" .
//
func (p *gcParser) parseMethodDecl() {
	// "func" already consumed
	scope := ast.NewScope(nil) // method scope
	p.expect('(')
	p.parseParameter(scope, nil) // receiver
	p.expect(')')
	p.expect(scanner.Ident)
	isVariadic := false
	p.parseSignature(scope, &isVariadic)

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
func (p *gcParser) parseExport() (string, *ast.Scope) {
	p.expectKeyword("package")
	name := p.expect(scanner.Ident)
	if p.tok != '\n' {
		// A package is safe if it was compiled with the -u flag,
		// which disables the unsafe package.
		// TODO(gri) remember "safe" package
		p.expectKeyword("safe")
	}
	p.expect('\n')

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

	return name, p.scope
}

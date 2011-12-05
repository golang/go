// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements an ast.Importer for gc generated object files.
// TODO(gri) Eventually move this into a separate package outside types.

package types

import (
	"errors"
	"fmt"
	"go/ast"
	"go/token"
	"io"
	"math/big"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"text/scanner"
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
		if f, err := os.Stat(filename); err == nil && !f.IsDir() {
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
	tok     rune                   // current token
	lit     string                 // literal string; only valid for Ident, Int, String tokens
	id      string                 // package id of imported package
	imports map[string]*ast.Object // package id -> package object
}

func (p *gcParser) init(filename, id string, src io.Reader, imports map[string]*ast.Object) {
	p.scanner.Init(src)
	p.scanner.Error = func(_ *scanner.Scanner, msg string) { p.error(msg) }
	p.scanner.Mode = scanner.ScanIdents | scanner.ScanInts | scanner.ScanStrings | scanner.ScanComments | scanner.SkipComments
	p.scanner.Whitespace = 1<<'\t' | 1<<' '
	p.scanner.Filename = filename // for good error messages
	p.next()
	p.id = id
	p.imports = imports
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
func GcImporter(imports map[string]*ast.Object, path string) (pkg *ast.Object, err error) {
	if path == "unsafe" {
		return Unsafe, nil
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
		err = errors.New("can't find import: " + id)
		return
	}

	if pkg = imports[id]; pkg != nil {
		return // package was imported before
	}

	buf, err := ExportData(filename)
	if err != nil {
		return
	}
	defer buf.Close()

	if trace {
		fmt.Printf("importing %s (%s)\n", id, filename)
	}

	var p gcParser
	p.init(filename, id, buf, imports)
	pkg = p.parseExport()
	return
}

// Declare inserts a named object of the given kind in scope.
func (p *gcParser) declare(scope *ast.Scope, kind ast.ObjKind, name string) *ast.Object {
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
		p.errorf("already declared: %v %s", kind, obj.Name)
	}

	// a new type object is a named type and may be referred
	// to before the underlying type is known - set it up
	if kind == ast.Typ {
		obj.Type = &Name{Obj: obj}
	}

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
		p.errorf("expected %q, got %q (%q)", scanner.TokenString(tok), scanner.TokenString(p.tok), lit)
	}
	p.next()
	return lit
}

func (p *gcParser) expectSpecial(tok string) {
	sep := rune('x') // not white space
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
func (p *gcParser) parsePkgId() *ast.Object {
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
		scope = ast.NewScope(nil)
		pkg = ast.NewObj(ast.Pkg, "")
		pkg.Data = scope
		p.imports[id] = pkg
	}

	return pkg
}

// dotIdentifier = ( ident | '·' ) { ident | int | '·' } .
func (p *gcParser) parseDotIdent() string {
	ident := ""
	if p.tok != scanner.Int {
		sep := rune('x') // not white space
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
func (p *gcParser) parseExportedName() (*ast.Object, string) {
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
	if obj == nil || obj.Kind != ast.Typ {
		p.errorf("not a basic type: %s", id)
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
	n, err := strconv.ParseUint(lit, 10, 64)
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
func (p *gcParser) parseName() (name string) {
	switch p.tok {
	case scanner.Ident:
		name = p.lit
		p.next()
	case '?':
		// anonymous
		p.next()
	case '@':
		// exported name prefixed with package path
		_, name = p.parseExportedName()
	default:
		p.error("name expected")
	}
	return
}

// Field = Name Type [ string_lit ] .
//
func (p *gcParser) parseField() (fld *ast.Object, tag string) {
	name := p.parseName()
	ftyp := p.parseType()
	if name == "" {
		// anonymous field - ftyp must be T or *T and T must be a type name
		if _, ok := Deref(ftyp).(*Name); !ok {
			p.errorf("anonymous field expected")
		}
	}
	if p.tok == scanner.String {
		tag = p.expect(scanner.String)
	}
	fld = ast.NewObj(ast.Var, name)
	fld.Type = ftyp
	return
}

// StructType = "struct" "{" [ FieldList ] "}" .
// FieldList  = Field { ";" Field } .
//
func (p *gcParser) parseStructType() Type {
	var fields []*ast.Object
	var tags []string

	parseField := func() {
		fld, tag := p.parseField()
		fields = append(fields, fld)
		tags = append(tags, tag)
	}

	p.expectKeyword("struct")
	p.expect('{')
	if p.tok != '}' {
		parseField()
		for p.tok == ';' {
			p.next()
			parseField()
		}
	}
	p.expect('}')

	return &Struct{Fields: fields, Tags: tags}
}

// Parameter = ( identifier | "?" ) [ "..." ] Type [ string_lit ] .
//
func (p *gcParser) parseParameter() (par *ast.Object, isVariadic bool) {
	name := p.parseName()
	if name == "" {
		name = "_" // cannot access unnamed identifiers
	}
	if p.tok == '.' {
		p.expectSpecial("...")
		isVariadic = true
	}
	ptyp := p.parseType()
	// ignore argument tag
	if p.tok == scanner.String {
		p.expect(scanner.String)
	}
	par = ast.NewObj(ast.Var, name)
	par.Type = ptyp
	return
}

// Parameters    = "(" [ ParameterList ] ")" .
// ParameterList = { Parameter "," } Parameter .
//
func (p *gcParser) parseParameters() (list []*ast.Object, isVariadic bool) {
	parseParameter := func() {
		par, variadic := p.parseParameter()
		list = append(list, par)
		if variadic {
			if isVariadic {
				p.error("... not on final argument")
			}
			isVariadic = true
		}
	}

	p.expect('(')
	if p.tok != ')' {
		parseParameter()
		for p.tok == ',' {
			p.next()
			parseParameter()
		}
	}
	p.expect(')')

	return
}

// Signature = Parameters [ Result ] .
// Result    = Type | Parameters .
//
func (p *gcParser) parseSignature() *Func {
	params, isVariadic := p.parseParameters()

	// optional result type
	var results []*ast.Object
	switch p.tok {
	case scanner.Ident, '[', '*', '<', '@':
		// single, unnamed result
		result := ast.NewObj(ast.Var, "_")
		result.Type = p.parseType()
		results = []*ast.Object{result}
	case '(':
		// named or multiple result(s)
		var variadic bool
		results, variadic = p.parseParameters()
		if variadic {
			p.error("... not permitted on result type")
		}
	}

	return &Func{Params: params, Results: results, IsVariadic: isVariadic}
}

// MethodSpec = ( identifier | ExportedName )  Signature .
//
func (p *gcParser) parseMethodSpec() *ast.Object {
	if p.tok == scanner.Ident {
		p.expect(scanner.Ident)
	} else {
		p.parseExportedName()
	}
	p.parseSignature()

	// TODO(gri) compute method object
	return ast.NewObj(ast.Fun, "_")
}

// InterfaceType = "interface" "{" [ MethodList ] "}" .
// MethodList    = MethodSpec { ";" MethodSpec } .
//
func (p *gcParser) parseInterfaceType() Type {
	var methods ObjList

	parseMethod := func() {
		meth := p.parseMethodSpec()
		methods = append(methods, meth)
	}

	p.expectKeyword("interface")
	p.expect('{')
	if p.tok != '}' {
		parseMethod()
		for p.tok == ';' {
			p.next()
			parseMethod()
		}
	}
	p.expect('}')

	methods.Sort()
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
// BasicType = ident .
// TypeName = ExportedName .
// SliceType = "[" "]" Type .
// PointerType = "*" Type .
// FuncType = "func" Signature .
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
		return p.declare(pkg.Data.(*ast.Scope), ast.Typ, name).Type.(Type)
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
		exp64, err := strconv.ParseUint(val, 10, 0)
		if err != nil {
			p.error(err)
		}
		exp := uint(exp64)
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
	pkg, name := p.parseExportedName()
	obj := p.declare(pkg.Data.(*ast.Scope), ast.Con, name)
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
	obj.Data = x
}

// TypeDecl = "type" ExportedName Type .
//
func (p *gcParser) parseTypeDecl() {
	p.expectKeyword("type")
	pkg, name := p.parseExportedName()
	obj := p.declare(pkg.Data.(*ast.Scope), ast.Typ, name)

	// The type object may have been imported before and thus already
	// have a type associated with it. We still need to parse the type
	// structure, but throw it away if the object already has a type.
	// This ensures that all imports refer to the same type object for
	// a given type declaration.
	typ := p.parseType()

	if name := obj.Type.(*Name); name.Underlying == nil {
		assert(Underlying(typ) == typ)
		name.Underlying = typ
	}
}

// VarDecl = "var" ExportedName Type .
//
func (p *gcParser) parseVarDecl() {
	p.expectKeyword("var")
	pkg, name := p.parseExportedName()
	obj := p.declare(pkg.Data.(*ast.Scope), ast.Var, name)
	obj.Type = p.parseType()
}

// FuncBody = "{" ... "}" .
// 
func (p *gcParser) parseFuncBody() {
	p.expect('{')
	for i := 1; i > 0; p.next() {
		switch p.tok {
		case '{':
			i++
		case '}':
			i--
		}
	}
}

// FuncDecl = "func" ExportedName Signature [ FuncBody ] .
//
func (p *gcParser) parseFuncDecl() {
	// "func" already consumed
	pkg, name := p.parseExportedName()
	obj := p.declare(pkg.Data.(*ast.Scope), ast.Fun, name)
	obj.Type = p.parseSignature()
	if p.tok == '{' {
		p.parseFuncBody()
	}
}

// MethodDecl = "func" Receiver Name Signature .
// Receiver   = "(" ( identifier | "?" ) [ "*" ] ExportedName ")" [ FuncBody ].
//
func (p *gcParser) parseMethodDecl() {
	// "func" already consumed
	p.expect('(')
	p.parseParameter() // receiver
	p.expect(')')
	p.parseName() // unexported method names in imports are qualified with their package.
	p.parseSignature()
	if p.tok == '{' {
		p.parseFuncBody()
	}
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
func (p *gcParser) parseExport() *ast.Object {
	p.expectKeyword("package")
	name := p.expect(scanner.Ident)
	if p.tok != '\n' {
		// A package is safe if it was compiled with the -u flag,
		// which disables the unsafe package.
		// TODO(gri) remember "safe" package
		p.expectKeyword("safe")
	}
	p.expect('\n')

	assert(p.imports[p.id] == nil)
	pkg := ast.NewObj(ast.Pkg, name)
	pkg.Data = ast.NewScope(nil)
	p.imports[p.id] = pkg

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

	return pkg
}

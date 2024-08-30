// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gccgoimporter

import (
	"errors"
	"fmt"
	"go/constant"
	"go/token"
	"go/types"
	"io"
	"strconv"
	"strings"
	"text/scanner"
	"unicode/utf8"
)

type parser struct {
	scanner  *scanner.Scanner
	version  string                    // format version
	tok      rune                      // current token
	lit      string                    // literal string; only valid for Ident, Int, String tokens
	pkgpath  string                    // package path of imported package
	pkgname  string                    // name of imported package
	pkg      *types.Package            // reference to imported package
	imports  map[string]*types.Package // package path -> package object
	typeList []types.Type              // type number -> type
	typeData []string                  // unparsed type data (v3 and later)
	fixups   []fixupRecord             // fixups to apply at end of parsing
	initdata InitData                  // package init priority data
	aliases  map[int]string            // maps saved type number to alias name
}

// When reading export data it's possible to encounter a defined type
// N1 with an underlying defined type N2 while we are still reading in
// that defined type N2; see issues #29006 and #29198 for instances
// of this. Example:
//
//   type N1 N2
//   type N2 struct {
//      ...
//      p *N1
//   }
//
// To handle such cases, the parser generates a fixup record (below) and
// delays setting of N1's underlying type until parsing is complete, at
// which point fixups are applied.

type fixupRecord struct {
	toUpdate *types.Named // type to modify when fixup is processed
	target   types.Type   // type that was incomplete when fixup was created
}

func (p *parser) init(filename string, src io.Reader, imports map[string]*types.Package) {
	p.scanner = new(scanner.Scanner)
	p.initScanner(filename, src)
	p.imports = imports
	p.aliases = make(map[int]string)
	p.typeList = make([]types.Type, 1 /* type numbers start at 1 */, 16)
}

func (p *parser) initScanner(filename string, src io.Reader) {
	p.scanner.Init(src)
	p.scanner.Error = func(_ *scanner.Scanner, msg string) { p.error(msg) }
	p.scanner.Mode = scanner.ScanIdents | scanner.ScanInts | scanner.ScanFloats | scanner.ScanStrings
	p.scanner.Whitespace = 1<<'\t' | 1<<' '
	p.scanner.Filename = filename // for good error messages
	p.next()
}

type importError struct {
	pos scanner.Position
	err error
}

func (e importError) Error() string {
	return fmt.Sprintf("import error %s (byte offset = %d): %s", e.pos, e.pos.Offset, e.err)
}

func (p *parser) error(err any) {
	if s, ok := err.(string); ok {
		err = errors.New(s)
	}
	// panic with a runtime.Error if err is not an error
	panic(importError{p.scanner.Pos(), err.(error)})
}

func (p *parser) errorf(format string, args ...any) {
	p.error(fmt.Errorf(format, args...))
}

func (p *parser) expect(tok rune) string {
	lit := p.lit
	if p.tok != tok {
		p.errorf("expected %s, got %s (%s)", scanner.TokenString(tok), scanner.TokenString(p.tok), lit)
	}
	p.next()
	return lit
}

func (p *parser) expectEOL() {
	if p.version == "v1" || p.version == "v2" {
		p.expect(';')
	}
	p.expect('\n')
}

func (p *parser) expectKeyword(keyword string) {
	lit := p.expect(scanner.Ident)
	if lit != keyword {
		p.errorf("expected keyword %s, got %q", keyword, lit)
	}
}

func (p *parser) parseString() string {
	str, err := strconv.Unquote(p.expect(scanner.String))
	if err != nil {
		p.error(err)
	}
	return str
}

// unquotedString     = { unquotedStringChar } .
// unquotedStringChar = <neither a whitespace nor a ';' char> .
func (p *parser) parseUnquotedString() string {
	if p.tok == scanner.EOF {
		p.error("unexpected EOF")
	}
	var b strings.Builder
	b.WriteString(p.scanner.TokenText())
	// This loop needs to examine each character before deciding whether to consume it. If we see a semicolon,
	// we need to let it be consumed by p.next().
	for ch := p.scanner.Peek(); ch != '\n' && ch != ';' && ch != scanner.EOF && p.scanner.Whitespace&(1<<uint(ch)) == 0; ch = p.scanner.Peek() {
		b.WriteRune(ch)
		p.scanner.Next()
	}
	p.next()
	return b.String()
}

func (p *parser) next() {
	p.tok = p.scanner.Scan()
	switch p.tok {
	case scanner.Ident, scanner.Int, scanner.Float, scanner.String, '·':
		p.lit = p.scanner.TokenText()
	default:
		p.lit = ""
	}
}

func (p *parser) parseQualifiedName() (path, name string) {
	return p.parseQualifiedNameStr(p.parseString())
}

func (p *parser) parseUnquotedQualifiedName() (path, name string) {
	return p.parseQualifiedNameStr(p.parseUnquotedString())
}

// qualifiedName = [ ["."] unquotedString "." ] unquotedString .
//
// The above production uses greedy matching.
func (p *parser) parseQualifiedNameStr(unquotedName string) (pkgpath, name string) {
	parts := strings.Split(unquotedName, ".")
	if parts[0] == "" {
		parts = parts[1:]
	}

	switch len(parts) {
	case 0:
		p.errorf("malformed qualified name: %q", unquotedName)
	case 1:
		// unqualified name
		pkgpath = p.pkgpath
		name = parts[0]
	default:
		// qualified name, which may contain periods
		pkgpath = strings.Join(parts[0:len(parts)-1], ".")
		name = parts[len(parts)-1]
	}

	return
}

// getPkg returns the package for a given path. If the package is
// not found but we have a package name, create the package and
// add it to the p.imports map.
func (p *parser) getPkg(pkgpath, name string) *types.Package {
	// package unsafe is not in the imports map - handle explicitly
	if pkgpath == "unsafe" {
		return types.Unsafe
	}
	pkg := p.imports[pkgpath]
	if pkg == nil && name != "" {
		pkg = types.NewPackage(pkgpath, name)
		p.imports[pkgpath] = pkg
	}
	return pkg
}

// parseExportedName is like parseQualifiedName, but
// the package path is resolved to an imported *types.Package.
//
// ExportedName = string [string] .
func (p *parser) parseExportedName() (pkg *types.Package, name string) {
	path, name := p.parseQualifiedName()
	var pkgname string
	if p.tok == scanner.String {
		pkgname = p.parseString()
	}
	pkg = p.getPkg(path, pkgname)
	if pkg == nil {
		p.errorf("package %s (path = %q) not found", name, path)
	}
	return
}

// Name = QualifiedName | "?" .
func (p *parser) parseName() string {
	if p.tok == '?' {
		// Anonymous.
		p.next()
		return ""
	}
	// The package path is redundant for us. Don't try to parse it.
	_, name := p.parseUnquotedQualifiedName()
	return name
}

func deref(typ types.Type) types.Type {
	if p, _ := typ.(*types.Pointer); p != nil {
		typ = p.Elem()
	}
	return typ
}

// Field = Name Type [string] .
func (p *parser) parseField(pkg *types.Package) (field *types.Var, tag string) {
	name := p.parseName()
	typ, n := p.parseTypeExtended(pkg)
	anon := false
	if name == "" {
		anon = true
		// Alias?
		if aname, ok := p.aliases[n]; ok {
			name = aname
		} else {
			switch typ := deref(typ).(type) {
			case *types.Basic:
				name = typ.Name()
			case *types.Named:
				name = typ.Obj().Name()
			default:
				p.error("embedded field expected")
			}
		}
	}
	field = types.NewField(token.NoPos, pkg, name, typ, anon)
	if p.tok == scanner.String {
		tag = p.parseString()
	}
	return
}

// Param = Name ["..."] Type .
func (p *parser) parseParam(pkg *types.Package) (param *types.Var, isVariadic bool) {
	name := p.parseName()
	// Ignore names invented for inlinable functions.
	if strings.HasPrefix(name, "p.") || strings.HasPrefix(name, "r.") || strings.HasPrefix(name, "$ret") {
		name = ""
	}
	if p.tok == '<' && p.scanner.Peek() == 'e' {
		// EscInfo = "<esc:" int ">" . (optional and ignored)
		p.next()
		p.expectKeyword("esc")
		p.expect(':')
		p.expect(scanner.Int)
		p.expect('>')
	}
	if p.tok == '.' {
		p.next()
		p.expect('.')
		p.expect('.')
		isVariadic = true
	}
	typ := p.parseType(pkg)
	if isVariadic {
		typ = types.NewSlice(typ)
	}
	param = types.NewParam(token.NoPos, pkg, name, typ)
	return
}

// Var = Name Type .
func (p *parser) parseVar(pkg *types.Package) *types.Var {
	name := p.parseName()
	v := types.NewVar(token.NoPos, pkg, name, p.parseType(pkg))
	if name[0] == '.' || name[0] == '<' {
		// This is an unexported variable,
		// or a variable defined in a different package.
		// We only want to record exported variables.
		return nil
	}
	return v
}

// Conversion = "convert" "(" Type "," ConstValue ")" .
func (p *parser) parseConversion(pkg *types.Package) (val constant.Value, typ types.Type) {
	p.expectKeyword("convert")
	p.expect('(')
	typ = p.parseType(pkg)
	p.expect(',')
	val, _ = p.parseConstValue(pkg)
	p.expect(')')
	return
}

// ConstValue     = string | "false" | "true" | ["-"] (int ["'"] | FloatOrComplex) | Conversion .
// FloatOrComplex = float ["i" | ("+"|"-") float "i"] .
func (p *parser) parseConstValue(pkg *types.Package) (val constant.Value, typ types.Type) {
	// v3 changed to $false, $true, $convert, to avoid confusion
	// with variable names in inline function bodies.
	if p.tok == '$' {
		p.next()
		if p.tok != scanner.Ident {
			p.errorf("expected identifier after '$', got %s (%q)", scanner.TokenString(p.tok), p.lit)
		}
	}

	switch p.tok {
	case scanner.String:
		str := p.parseString()
		val = constant.MakeString(str)
		typ = types.Typ[types.UntypedString]
		return

	case scanner.Ident:
		b := false
		switch p.lit {
		case "false":
		case "true":
			b = true

		case "convert":
			return p.parseConversion(pkg)

		default:
			p.errorf("expected const value, got %s (%q)", scanner.TokenString(p.tok), p.lit)
		}

		p.next()
		val = constant.MakeBool(b)
		typ = types.Typ[types.UntypedBool]
		return
	}

	sign := ""
	if p.tok == '-' {
		p.next()
		sign = "-"
	}

	switch p.tok {
	case scanner.Int:
		val = constant.MakeFromLiteral(sign+p.lit, token.INT, 0)
		if val == nil {
			p.error("could not parse integer literal")
		}

		p.next()
		if p.tok == '\'' {
			p.next()
			typ = types.Typ[types.UntypedRune]
		} else {
			typ = types.Typ[types.UntypedInt]
		}

	case scanner.Float:
		re := sign + p.lit
		p.next()

		var im string
		switch p.tok {
		case '+':
			p.next()
			im = p.expect(scanner.Float)

		case '-':
			p.next()
			im = "-" + p.expect(scanner.Float)

		case scanner.Ident:
			// re is in fact the imaginary component. Expect "i" below.
			im = re
			re = "0"

		default:
			val = constant.MakeFromLiteral(re, token.FLOAT, 0)
			if val == nil {
				p.error("could not parse float literal")
			}
			typ = types.Typ[types.UntypedFloat]
			return
		}

		p.expectKeyword("i")
		reval := constant.MakeFromLiteral(re, token.FLOAT, 0)
		if reval == nil {
			p.error("could not parse real component of complex literal")
		}
		imval := constant.MakeFromLiteral(im+"i", token.IMAG, 0)
		if imval == nil {
			p.error("could not parse imag component of complex literal")
		}
		val = constant.BinaryOp(reval, token.ADD, imval)
		typ = types.Typ[types.UntypedComplex]

	default:
		p.errorf("expected const value, got %s (%q)", scanner.TokenString(p.tok), p.lit)
	}

	return
}

// Const = Name [Type] "=" ConstValue .
func (p *parser) parseConst(pkg *types.Package) *types.Const {
	name := p.parseName()
	var typ types.Type
	if p.tok == '<' {
		typ = p.parseType(pkg)
	}
	p.expect('=')
	val, vtyp := p.parseConstValue(pkg)
	if typ == nil {
		typ = vtyp
	}
	return types.NewConst(token.NoPos, pkg, name, typ, val)
}

// reserved is a singleton type used to fill type map slots that have
// been reserved (i.e., for which a type number has been parsed) but
// which don't have their actual type yet. When the type map is updated,
// the actual type must replace a reserved entry (or we have an internal
// error). Used for self-verification only - not required for correctness.
var reserved = new(struct{ types.Type })

// reserve reserves the type map entry n for future use.
func (p *parser) reserve(n int) {
	// Notes:
	// - for pre-V3 export data, the type numbers we see are
	//   guaranteed to be in increasing order, so we append a
	//   reserved entry onto the list.
	// - for V3+ export data, type numbers can appear in
	//   any order, however the 'types' section tells us the
	//   total number of types, hence typeList is pre-allocated.
	if len(p.typeData) == 0 {
		if n != len(p.typeList) {
			p.errorf("invalid type number %d (out of sync)", n)
		}
		p.typeList = append(p.typeList, reserved)
	} else {
		if p.typeList[n] != nil {
			p.errorf("previously visited type number %d", n)
		}
		p.typeList[n] = reserved
	}
}

// update sets the type map entries for the entries in nlist to t.
// An entry in nlist can be a type number in p.typeList,
// used to resolve named types, or it can be a *types.Pointer,
// used to resolve pointers to named types in case they are referenced
// by embedded fields.
func (p *parser) update(t types.Type, nlist []any) {
	if t == reserved {
		p.errorf("internal error: update(%v) invoked on reserved", nlist)
	}
	if t == nil {
		p.errorf("internal error: update(%v) invoked on nil", nlist)
	}
	for _, n := range nlist {
		switch n := n.(type) {
		case int:
			if p.typeList[n] == t {
				continue
			}
			if p.typeList[n] != reserved {
				p.errorf("internal error: update(%v): %d not reserved", nlist, n)
			}
			p.typeList[n] = t
		case *types.Pointer:
			if *n != (types.Pointer{}) {
				elem := n.Elem()
				if elem == t {
					continue
				}
				p.errorf("internal error: update: pointer already set to %v, expected %v", elem, t)
			}
			*n = *types.NewPointer(t)
		default:
			p.errorf("internal error: %T on nlist", n)
		}
	}
}

// NamedType = TypeName [ "=" ] Type { Method } .
// TypeName  = ExportedName .
// Method    = "func" "(" Param ")" Name ParamList ResultList [InlineBody] ";" .
func (p *parser) parseNamedType(nlist []any) types.Type {
	pkg, name := p.parseExportedName()
	scope := pkg.Scope()
	obj := scope.Lookup(name)
	if obj != nil && obj.Type() == nil {
		p.errorf("%v has nil type", obj)
	}

	if p.tok == scanner.Ident && p.lit == "notinheap" {
		p.next()
		// The go/types package has no way of recording that
		// this type is marked notinheap. Presumably no user
		// of this package actually cares.
	}

	// type alias
	if p.tok == '=' {
		p.next()
		p.aliases[nlist[len(nlist)-1].(int)] = name
		if obj != nil {
			// use the previously imported (canonical) type
			t := obj.Type()
			p.update(t, nlist)
			p.parseType(pkg) // discard
			return t
		}
		t := p.parseType(pkg, nlist...)
		obj = types.NewTypeName(token.NoPos, pkg, name, t)
		scope.Insert(obj)
		return t
	}

	// defined type
	if obj == nil {
		// A named type may be referred to before the underlying type
		// is known - set it up.
		tname := types.NewTypeName(token.NoPos, pkg, name, nil)
		types.NewNamed(tname, nil, nil)
		scope.Insert(tname)
		obj = tname
	}

	// use the previously imported (canonical), or newly created type
	t := obj.Type()
	p.update(t, nlist)

	nt, ok := t.(*types.Named)
	if !ok {
		// This can happen for unsafe.Pointer, which is a TypeName holding a Basic type.
		pt := p.parseType(pkg)
		if pt != t {
			p.error("unexpected underlying type for non-named TypeName")
		}
		return t
	}

	underlying := p.parseType(pkg)
	if nt.Underlying() == nil {
		if underlying.Underlying() == nil {
			fix := fixupRecord{toUpdate: nt, target: underlying}
			p.fixups = append(p.fixups, fix)
		} else {
			nt.SetUnderlying(underlying.Underlying())
		}
	}

	if p.tok == '\n' {
		p.next()
		// collect associated methods
		for p.tok == scanner.Ident {
			p.expectKeyword("func")
			if p.tok == '/' {
				// Skip a /*nointerface*/ or /*asm ID */ comment.
				p.expect('/')
				p.expect('*')
				if p.expect(scanner.Ident) == "asm" {
					p.parseUnquotedString()
				}
				p.expect('*')
				p.expect('/')
			}
			p.expect('(')
			receiver, _ := p.parseParam(pkg)
			p.expect(')')
			name := p.parseName()
			params, isVariadic := p.parseParamList(pkg)
			results := p.parseResultList(pkg)
			p.skipInlineBody()
			p.expectEOL()

			sig := types.NewSignatureType(receiver, nil, nil, params, results, isVariadic)
			nt.AddMethod(types.NewFunc(token.NoPos, pkg, name, sig))
		}
	}

	return nt
}

func (p *parser) parseInt64() int64 {
	lit := p.expect(scanner.Int)
	n, err := strconv.ParseInt(lit, 10, 64)
	if err != nil {
		p.error(err)
	}
	return n
}

func (p *parser) parseInt() int {
	lit := p.expect(scanner.Int)
	n, err := strconv.ParseInt(lit, 10, 0 /* int */)
	if err != nil {
		p.error(err)
	}
	return int(n)
}

// ArrayOrSliceType = "[" [ int ] "]" Type .
func (p *parser) parseArrayOrSliceType(pkg *types.Package, nlist []any) types.Type {
	p.expect('[')
	if p.tok == ']' {
		p.next()

		t := new(types.Slice)
		p.update(t, nlist)

		*t = *types.NewSlice(p.parseType(pkg))
		return t
	}

	t := new(types.Array)
	p.update(t, nlist)

	len := p.parseInt64()
	p.expect(']')

	*t = *types.NewArray(p.parseType(pkg), len)
	return t
}

// MapType = "map" "[" Type "]" Type .
func (p *parser) parseMapType(pkg *types.Package, nlist []any) types.Type {
	p.expectKeyword("map")

	t := new(types.Map)
	p.update(t, nlist)

	p.expect('[')
	key := p.parseType(pkg)
	p.expect(']')
	elem := p.parseType(pkg)

	*t = *types.NewMap(key, elem)
	return t
}

// ChanType = "chan" ["<-" | "-<"] Type .
func (p *parser) parseChanType(pkg *types.Package, nlist []any) types.Type {
	p.expectKeyword("chan")

	t := new(types.Chan)
	p.update(t, nlist)

	dir := types.SendRecv
	switch p.tok {
	case '-':
		p.next()
		p.expect('<')
		dir = types.SendOnly

	case '<':
		// don't consume '<' if it belongs to Type
		if p.scanner.Peek() == '-' {
			p.next()
			p.expect('-')
			dir = types.RecvOnly
		}
	}

	*t = *types.NewChan(dir, p.parseType(pkg))
	return t
}

// StructType = "struct" "{" { Field } "}" .
func (p *parser) parseStructType(pkg *types.Package, nlist []any) types.Type {
	p.expectKeyword("struct")

	t := new(types.Struct)
	p.update(t, nlist)

	var fields []*types.Var
	var tags []string

	p.expect('{')
	for p.tok != '}' && p.tok != scanner.EOF {
		field, tag := p.parseField(pkg)
		p.expect(';')
		fields = append(fields, field)
		tags = append(tags, tag)
	}
	p.expect('}')

	*t = *types.NewStruct(fields, tags)
	return t
}

// ParamList = "(" [ { Parameter "," } Parameter ] ")" .
func (p *parser) parseParamList(pkg *types.Package) (*types.Tuple, bool) {
	var list []*types.Var
	isVariadic := false

	p.expect('(')
	for p.tok != ')' && p.tok != scanner.EOF {
		if len(list) > 0 {
			p.expect(',')
		}
		par, variadic := p.parseParam(pkg)
		list = append(list, par)
		if variadic {
			if isVariadic {
				p.error("... not on final argument")
			}
			isVariadic = true
		}
	}
	p.expect(')')

	return types.NewTuple(list...), isVariadic
}

// ResultList = Type | ParamList .
func (p *parser) parseResultList(pkg *types.Package) *types.Tuple {
	switch p.tok {
	case '<':
		p.next()
		if p.tok == scanner.Ident && p.lit == "inl" {
			return nil
		}
		taa, _ := p.parseTypeAfterAngle(pkg)
		return types.NewTuple(types.NewParam(token.NoPos, pkg, "", taa))

	case '(':
		params, _ := p.parseParamList(pkg)
		return params

	default:
		return nil
	}
}

// FunctionType = ParamList ResultList .
func (p *parser) parseFunctionType(pkg *types.Package, nlist []any) *types.Signature {
	t := new(types.Signature)
	p.update(t, nlist)

	params, isVariadic := p.parseParamList(pkg)
	results := p.parseResultList(pkg)

	*t = *types.NewSignatureType(nil, nil, nil, params, results, isVariadic)
	return t
}

// Func = Name FunctionType [InlineBody] .
func (p *parser) parseFunc(pkg *types.Package) *types.Func {
	if p.tok == '/' {
		// Skip an /*asm ID */ comment.
		p.expect('/')
		p.expect('*')
		if p.expect(scanner.Ident) == "asm" {
			p.parseUnquotedString()
		}
		p.expect('*')
		p.expect('/')
	}

	name := p.parseName()
	f := types.NewFunc(token.NoPos, pkg, name, p.parseFunctionType(pkg, nil))
	p.skipInlineBody()

	if name[0] == '.' || name[0] == '<' || strings.ContainsRune(name, '$') {
		// This is an unexported function,
		// or a function defined in a different package,
		// or a type$equal or type$hash function.
		// We only want to record exported functions.
		return nil
	}

	return f
}

// InterfaceType = "interface" "{" { ("?" Type | Func) ";" } "}" .
func (p *parser) parseInterfaceType(pkg *types.Package, nlist []any) types.Type {
	p.expectKeyword("interface")

	t := new(types.Interface)
	p.update(t, nlist)

	var methods []*types.Func
	var embeddeds []types.Type

	p.expect('{')
	for p.tok != '}' && p.tok != scanner.EOF {
		if p.tok == '?' {
			p.next()
			embeddeds = append(embeddeds, p.parseType(pkg))
		} else {
			method := p.parseFunc(pkg)
			if method != nil {
				methods = append(methods, method)
			}
		}
		p.expect(';')
	}
	p.expect('}')

	*t = *types.NewInterfaceType(methods, embeddeds)
	return t
}

// PointerType = "*" ("any" | Type) .
func (p *parser) parsePointerType(pkg *types.Package, nlist []any) types.Type {
	p.expect('*')
	if p.tok == scanner.Ident {
		p.expectKeyword("any")
		t := types.Typ[types.UnsafePointer]
		p.update(t, nlist)
		return t
	}

	t := new(types.Pointer)
	p.update(t, nlist)

	*t = *types.NewPointer(p.parseType(pkg, t))

	return t
}

// TypeSpec = NamedType | MapType | ChanType | StructType | InterfaceType | PointerType | ArrayOrSliceType | FunctionType .
func (p *parser) parseTypeSpec(pkg *types.Package, nlist []any) types.Type {
	switch p.tok {
	case scanner.String:
		return p.parseNamedType(nlist)

	case scanner.Ident:
		switch p.lit {
		case "map":
			return p.parseMapType(pkg, nlist)

		case "chan":
			return p.parseChanType(pkg, nlist)

		case "struct":
			return p.parseStructType(pkg, nlist)

		case "interface":
			return p.parseInterfaceType(pkg, nlist)
		}

	case '*':
		return p.parsePointerType(pkg, nlist)

	case '[':
		return p.parseArrayOrSliceType(pkg, nlist)

	case '(':
		return p.parseFunctionType(pkg, nlist)
	}

	p.errorf("expected type name or literal, got %s", scanner.TokenString(p.tok))
	return nil
}

const (
	// From gofrontend/go/export.h
	// Note that these values are negative in the gofrontend and have been made positive
	// in the gccgoimporter.
	gccgoBuiltinINT8       = 1
	gccgoBuiltinINT16      = 2
	gccgoBuiltinINT32      = 3
	gccgoBuiltinINT64      = 4
	gccgoBuiltinUINT8      = 5
	gccgoBuiltinUINT16     = 6
	gccgoBuiltinUINT32     = 7
	gccgoBuiltinUINT64     = 8
	gccgoBuiltinFLOAT32    = 9
	gccgoBuiltinFLOAT64    = 10
	gccgoBuiltinINT        = 11
	gccgoBuiltinUINT       = 12
	gccgoBuiltinUINTPTR    = 13
	gccgoBuiltinBOOL       = 15
	gccgoBuiltinSTRING     = 16
	gccgoBuiltinCOMPLEX64  = 17
	gccgoBuiltinCOMPLEX128 = 18
	gccgoBuiltinERROR      = 19
	gccgoBuiltinBYTE       = 20
	gccgoBuiltinRUNE       = 21
	gccgoBuiltinANY        = 22
)

func lookupBuiltinType(typ int) types.Type {
	return [...]types.Type{
		gccgoBuiltinINT8:       types.Typ[types.Int8],
		gccgoBuiltinINT16:      types.Typ[types.Int16],
		gccgoBuiltinINT32:      types.Typ[types.Int32],
		gccgoBuiltinINT64:      types.Typ[types.Int64],
		gccgoBuiltinUINT8:      types.Typ[types.Uint8],
		gccgoBuiltinUINT16:     types.Typ[types.Uint16],
		gccgoBuiltinUINT32:     types.Typ[types.Uint32],
		gccgoBuiltinUINT64:     types.Typ[types.Uint64],
		gccgoBuiltinFLOAT32:    types.Typ[types.Float32],
		gccgoBuiltinFLOAT64:    types.Typ[types.Float64],
		gccgoBuiltinINT:        types.Typ[types.Int],
		gccgoBuiltinUINT:       types.Typ[types.Uint],
		gccgoBuiltinUINTPTR:    types.Typ[types.Uintptr],
		gccgoBuiltinBOOL:       types.Typ[types.Bool],
		gccgoBuiltinSTRING:     types.Typ[types.String],
		gccgoBuiltinCOMPLEX64:  types.Typ[types.Complex64],
		gccgoBuiltinCOMPLEX128: types.Typ[types.Complex128],
		gccgoBuiltinERROR:      types.Universe.Lookup("error").Type(),
		gccgoBuiltinBYTE:       types.Universe.Lookup("byte").Type(),
		gccgoBuiltinRUNE:       types.Universe.Lookup("rune").Type(),
		gccgoBuiltinANY:        types.Universe.Lookup("any").Type(),
	}[typ]
}

// Type = "<" "type" ( "-" int | int [ TypeSpec ] ) ">" .
//
// parseType updates the type map to t for all type numbers n.
func (p *parser) parseType(pkg *types.Package, n ...any) types.Type {
	p.expect('<')
	t, _ := p.parseTypeAfterAngle(pkg, n...)
	return t
}

// (*parser).Type after reading the "<".
func (p *parser) parseTypeAfterAngle(pkg *types.Package, n ...any) (t types.Type, n1 int) {
	p.expectKeyword("type")

	n1 = 0
	switch p.tok {
	case scanner.Int:
		n1 = p.parseInt()
		if p.tok == '>' {
			if len(p.typeData) > 0 && p.typeList[n1] == nil {
				p.parseSavedType(pkg, n1, n)
			}
			t = p.typeList[n1]
			if len(p.typeData) == 0 && t == reserved {
				p.errorf("invalid type cycle, type %d not yet defined (nlist=%v)", n1, n)
			}
			p.update(t, n)
		} else {
			p.reserve(n1)
			t = p.parseTypeSpec(pkg, append(n, n1))
		}

	case '-':
		p.next()
		n1 := p.parseInt()
		t = lookupBuiltinType(n1)
		p.update(t, n)

	default:
		p.errorf("expected type number, got %s (%q)", scanner.TokenString(p.tok), p.lit)
		return nil, 0
	}

	if t == nil || t == reserved {
		p.errorf("internal error: bad return from parseType(%v)", n)
	}

	p.expect('>')
	return
}

// parseTypeExtended is identical to parseType, but if the type in
// question is a saved type, returns the index as well as the type
// pointer (index returned is zero if we parsed a builtin).
func (p *parser) parseTypeExtended(pkg *types.Package, n ...any) (t types.Type, n1 int) {
	p.expect('<')
	t, n1 = p.parseTypeAfterAngle(pkg, n...)
	return
}

// InlineBody = "<inl:NN>" .{NN}
// Reports whether a body was skipped.
func (p *parser) skipInlineBody() {
	// We may or may not have seen the '<' already, depending on
	// whether the function had a result type or not.
	if p.tok == '<' {
		p.next()
		p.expectKeyword("inl")
	} else if p.tok != scanner.Ident || p.lit != "inl" {
		return
	} else {
		p.next()
	}

	p.expect(':')
	want := p.parseInt()
	p.expect('>')

	defer func(w uint64) {
		p.scanner.Whitespace = w
	}(p.scanner.Whitespace)
	p.scanner.Whitespace = 0

	got := 0
	for got < want {
		r := p.scanner.Next()
		if r == scanner.EOF {
			p.error("unexpected EOF")
		}
		got += utf8.RuneLen(r)
	}
}

// Types = "types" maxp1 exportedp1 (offset length)* .
func (p *parser) parseTypes(pkg *types.Package) {
	maxp1 := p.parseInt()
	exportedp1 := p.parseInt()
	p.typeList = make([]types.Type, maxp1, maxp1)

	type typeOffset struct {
		offset int
		length int
	}
	var typeOffsets []typeOffset

	total := 0
	for i := 1; i < maxp1; i++ {
		len := p.parseInt()
		typeOffsets = append(typeOffsets, typeOffset{total, len})
		total += len
	}

	defer func(w uint64) {
		p.scanner.Whitespace = w
	}(p.scanner.Whitespace)
	p.scanner.Whitespace = 0

	// We should now have p.tok pointing to the final newline.
	// The next runes from the scanner should be the type data.

	var sb strings.Builder
	for sb.Len() < total {
		r := p.scanner.Next()
		if r == scanner.EOF {
			p.error("unexpected EOF")
		}
		sb.WriteRune(r)
	}
	allTypeData := sb.String()

	p.typeData = []string{""} // type 0, unused
	for _, to := range typeOffsets {
		p.typeData = append(p.typeData, allTypeData[to.offset:to.offset+to.length])
	}

	for i := 1; i < exportedp1; i++ {
		p.parseSavedType(pkg, i, nil)
	}
}

// parseSavedType parses one saved type definition.
func (p *parser) parseSavedType(pkg *types.Package, i int, nlist []any) {
	defer func(s *scanner.Scanner, tok rune, lit string) {
		p.scanner = s
		p.tok = tok
		p.lit = lit
	}(p.scanner, p.tok, p.lit)

	p.scanner = new(scanner.Scanner)
	p.initScanner(p.scanner.Filename, strings.NewReader(p.typeData[i]))
	p.expectKeyword("type")
	id := p.parseInt()
	if id != i {
		p.errorf("type ID mismatch: got %d, want %d", id, i)
	}
	if p.typeList[i] == reserved {
		p.errorf("internal error: %d already reserved in parseSavedType", i)
	}
	if p.typeList[i] == nil {
		p.reserve(i)
		p.parseTypeSpec(pkg, append(nlist, i))
	}
	if p.typeList[i] == nil || p.typeList[i] == reserved {
		p.errorf("internal error: parseSavedType(%d,%v) reserved/nil", i, nlist)
	}
}

// PackageInit = unquotedString unquotedString int .
func (p *parser) parsePackageInit() PackageInit {
	name := p.parseUnquotedString()
	initfunc := p.parseUnquotedString()
	priority := -1
	if p.version == "v1" {
		priority = p.parseInt()
	}
	return PackageInit{Name: name, InitFunc: initfunc, Priority: priority}
}

// Create the package if we have parsed both the package path and package name.
func (p *parser) maybeCreatePackage() {
	if p.pkgname != "" && p.pkgpath != "" {
		p.pkg = p.getPkg(p.pkgpath, p.pkgname)
	}
}

// InitDataDirective = ( "v1" | "v2" | "v3" ) ";" |
//
//	"priority" int ";" |
//	"init" { PackageInit } ";" |
//	"checksum" unquotedString ";" .
func (p *parser) parseInitDataDirective() {
	if p.tok != scanner.Ident {
		// unexpected token kind; panic
		p.expect(scanner.Ident)
	}

	switch p.lit {
	case "v1", "v2", "v3":
		p.version = p.lit
		p.next()
		p.expect(';')
		p.expect('\n')

	case "priority":
		p.next()
		p.initdata.Priority = p.parseInt()
		p.expectEOL()

	case "init":
		p.next()
		for p.tok != '\n' && p.tok != ';' && p.tok != scanner.EOF {
			p.initdata.Inits = append(p.initdata.Inits, p.parsePackageInit())
		}
		p.expectEOL()

	case "init_graph":
		p.next()
		// The graph data is thrown away for now.
		for p.tok != '\n' && p.tok != ';' && p.tok != scanner.EOF {
			p.parseInt64()
			p.parseInt64()
		}
		p.expectEOL()

	case "checksum":
		// Don't let the scanner try to parse the checksum as a number.
		defer func(mode uint) {
			p.scanner.Mode = mode
		}(p.scanner.Mode)
		p.scanner.Mode &^= scanner.ScanInts | scanner.ScanFloats
		p.next()
		p.parseUnquotedString()
		p.expectEOL()

	default:
		p.errorf("unexpected identifier: %q", p.lit)
	}
}

// Directive = InitDataDirective |
//
//	"package" unquotedString [ unquotedString ] [ unquotedString ] ";" |
//	"pkgpath" unquotedString ";" |
//	"prefix" unquotedString ";" |
//	"import" unquotedString unquotedString string ";" |
//	"indirectimport" unquotedString unquotedstring ";" |
//	"func" Func ";" |
//	"type" Type ";" |
//	"var" Var ";" |
//	"const" Const ";" .
func (p *parser) parseDirective() {
	if p.tok != scanner.Ident {
		// unexpected token kind; panic
		p.expect(scanner.Ident)
	}

	switch p.lit {
	case "v1", "v2", "v3", "priority", "init", "init_graph", "checksum":
		p.parseInitDataDirective()

	case "package":
		p.next()
		p.pkgname = p.parseUnquotedString()
		p.maybeCreatePackage()
		if p.version != "v1" && p.tok != '\n' && p.tok != ';' {
			p.parseUnquotedString()
			p.parseUnquotedString()
		}
		p.expectEOL()

	case "pkgpath":
		p.next()
		p.pkgpath = p.parseUnquotedString()
		p.maybeCreatePackage()
		p.expectEOL()

	case "prefix":
		p.next()
		p.pkgpath = p.parseUnquotedString()
		p.expectEOL()

	case "import":
		p.next()
		pkgname := p.parseUnquotedString()
		pkgpath := p.parseUnquotedString()
		p.getPkg(pkgpath, pkgname)
		p.parseString()
		p.expectEOL()

	case "indirectimport":
		p.next()
		pkgname := p.parseUnquotedString()
		pkgpath := p.parseUnquotedString()
		p.getPkg(pkgpath, pkgname)
		p.expectEOL()

	case "types":
		p.next()
		p.parseTypes(p.pkg)
		p.expectEOL()

	case "func":
		p.next()
		fun := p.parseFunc(p.pkg)
		if fun != nil {
			p.pkg.Scope().Insert(fun)
		}
		p.expectEOL()

	case "type":
		p.next()
		p.parseType(p.pkg)
		p.expectEOL()

	case "var":
		p.next()
		v := p.parseVar(p.pkg)
		if v != nil {
			p.pkg.Scope().Insert(v)
		}
		p.expectEOL()

	case "const":
		p.next()
		c := p.parseConst(p.pkg)
		p.pkg.Scope().Insert(c)
		p.expectEOL()

	default:
		p.errorf("unexpected identifier: %q", p.lit)
	}
}

// Package = { Directive } .
func (p *parser) parsePackage() *types.Package {
	for p.tok != scanner.EOF {
		p.parseDirective()
	}
	for _, f := range p.fixups {
		if f.target.Underlying() == nil {
			p.errorf("internal error: fixup can't be applied, loop required")
		}
		f.toUpdate.SetUnderlying(f.target.Underlying())
	}
	p.fixups = nil
	for _, typ := range p.typeList {
		if it, ok := typ.(*types.Interface); ok {
			it.Complete()
		}
	}
	p.pkg.MarkComplete()
	return p.pkg
}

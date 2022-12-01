// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file is a modified copy of $GOROOT/src/go/internal/gcimporter/gcimporter.go,
// but it also contains the original source-based importer code for Go1.6.
// Once we stop supporting 1.6, we can remove that code.

// Package gcimporter provides various functions for reading
// gc-generated object files that can be used to implement the
// Importer interface defined by the Go 1.5 standard library package.
package gcimporter // import "golang.org/x/tools/internal/gcimporter"

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"go/build"
	"go/constant"
	"go/token"
	"go/types"
	"io"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"text/scanner"
)

const (
	// Enable debug during development: it adds some additional checks, and
	// prevents errors from being recovered.
	debug = false

	// If trace is set, debugging output is printed to std out.
	trace = false
)

var exportMap sync.Map // package dir → func() (string, bool)

// lookupGorootExport returns the location of the export data
// (normally found in the build cache, but located in GOROOT/pkg
// in prior Go releases) for the package located in pkgDir.
//
// (We use the package's directory instead of its import path
// mainly to simplify handling of the packages in src/vendor
// and cmd/vendor.)
func lookupGorootExport(pkgDir string) (string, bool) {
	f, ok := exportMap.Load(pkgDir)
	if !ok {
		var (
			listOnce   sync.Once
			exportPath string
		)
		f, _ = exportMap.LoadOrStore(pkgDir, func() (string, bool) {
			listOnce.Do(func() {
				cmd := exec.Command("go", "list", "-export", "-f", "{{.Export}}", pkgDir)
				cmd.Dir = build.Default.GOROOT
				var output []byte
				output, err := cmd.Output()
				if err != nil {
					return
				}

				exports := strings.Split(string(bytes.TrimSpace(output)), "\n")
				if len(exports) != 1 {
					return
				}

				exportPath = exports[0]
			})

			return exportPath, exportPath != ""
		})
	}

	return f.(func() (string, bool))()
}

var pkgExts = [...]string{".a", ".o"}

// FindPkg returns the filename and unique package id for an import
// path based on package information provided by build.Import (using
// the build.Default build.Context). A relative srcDir is interpreted
// relative to the current working directory.
// If no file was found, an empty filename is returned.
func FindPkg(path, srcDir string) (filename, id string) {
	if path == "" {
		return
	}

	var noext string
	switch {
	default:
		// "x" -> "$GOPATH/pkg/$GOOS_$GOARCH/x.ext", "x"
		// Don't require the source files to be present.
		if abs, err := filepath.Abs(srcDir); err == nil { // see issue 14282
			srcDir = abs
		}
		bp, _ := build.Import(path, srcDir, build.FindOnly|build.AllowBinary)
		if bp.PkgObj == "" {
			var ok bool
			if bp.Goroot && bp.Dir != "" {
				filename, ok = lookupGorootExport(bp.Dir)
			}
			if !ok {
				id = path // make sure we have an id to print in error message
				return
			}
		} else {
			noext = strings.TrimSuffix(bp.PkgObj, ".a")
			id = bp.ImportPath
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
		id = path
	}

	if false { // for debugging
		if path != id {
			fmt.Printf("%s -> %s\n", path, id)
		}
	}

	if filename != "" {
		if f, err := os.Stat(filename); err == nil && !f.IsDir() {
			return
		}
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

// ImportData imports a package by reading the gc-generated export data,
// adds the corresponding package object to the packages map indexed by id,
// and returns the object.
//
// The packages map must contains all packages already imported. The data
// reader position must be the beginning of the export data section. The
// filename is only used in error messages.
//
// If packages[id] contains the completely imported package, that package
// can be used directly, and there is no need to call this function (but
// there is also no harm but for extra time used).
func ImportData(packages map[string]*types.Package, filename, id string, data io.Reader) (pkg *types.Package, err error) {
	// support for parser error handling
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

	var p parser
	p.init(filename, id, data, packages)
	pkg = p.parseExport()

	return
}

// Import imports a gc-generated package given its import path and srcDir, adds
// the corresponding package object to the packages map, and returns the object.
// The packages map must contain all packages already imported.
func Import(packages map[string]*types.Package, path, srcDir string, lookup func(path string) (io.ReadCloser, error)) (pkg *types.Package, err error) {
	var rc io.ReadCloser
	var filename, id string
	if lookup != nil {
		// With custom lookup specified, assume that caller has
		// converted path to a canonical import path for use in the map.
		if path == "unsafe" {
			return types.Unsafe, nil
		}
		id = path

		// No need to re-import if the package was imported completely before.
		if pkg = packages[id]; pkg != nil && pkg.Complete() {
			return
		}
		f, err := lookup(path)
		if err != nil {
			return nil, err
		}
		rc = f
	} else {
		filename, id = FindPkg(path, srcDir)
		if filename == "" {
			if path == "unsafe" {
				return types.Unsafe, nil
			}
			return nil, fmt.Errorf("can't find import: %q", id)
		}

		// no need to re-import if the package was imported completely before
		if pkg = packages[id]; pkg != nil && pkg.Complete() {
			return
		}

		// open file
		f, err := os.Open(filename)
		if err != nil {
			return nil, err
		}
		defer func() {
			if err != nil {
				// add file name to error
				err = fmt.Errorf("%s: %v", filename, err)
			}
		}()
		rc = f
	}
	defer rc.Close()

	var hdr string
	var size int64
	buf := bufio.NewReader(rc)
	if hdr, size, err = FindExportData(buf); err != nil {
		return
	}

	switch hdr {
	case "$$\n":
		// Work-around if we don't have a filename; happens only if lookup != nil.
		// Either way, the filename is only needed for importer error messages, so
		// this is fine.
		if filename == "" {
			filename = path
		}
		return ImportData(packages, filename, id, buf)

	case "$$B\n":
		var data []byte
		data, err = ioutil.ReadAll(buf)
		if err != nil {
			break
		}

		// TODO(gri): allow clients of go/importer to provide a FileSet.
		// Or, define a new standard go/types/gcexportdata package.
		fset := token.NewFileSet()

		// The indexed export format starts with an 'i'; the older
		// binary export format starts with a 'c', 'd', or 'v'
		// (from "version"). Select appropriate importer.
		if len(data) > 0 {
			switch data[0] {
			case 'i':
				_, pkg, err := IImportData(fset, packages, data[1:], id)
				return pkg, err

			case 'v', 'c', 'd':
				_, pkg, err := BImportData(fset, packages, data, id)
				return pkg, err

			case 'u':
				_, pkg, err := UImportData(fset, packages, data[1:size], id)
				return pkg, err

			default:
				l := len(data)
				if l > 10 {
					l = 10
				}
				return nil, fmt.Errorf("unexpected export data with prefix %q for path %s", string(data[:l]), id)
			}
		}

	default:
		err = fmt.Errorf("unknown export data header: %q", hdr)
	}

	return
}

// ----------------------------------------------------------------------------
// Parser

// TODO(gri) Imported objects don't have position information.
//           Ideally use the debug table line info; alternatively
//           create some fake position (or the position of the
//           import). That way error messages referring to imported
//           objects can print meaningful information.

// parser parses the exports inside a gc compiler-produced
// object/archive file and populates its scope with the results.
type parser struct {
	scanner    scanner.Scanner
	tok        rune                      // current token
	lit        string                    // literal string; only valid for Ident, Int, String tokens
	id         string                    // package id of imported package
	sharedPkgs map[string]*types.Package // package id -> package object (across importer)
	localPkgs  map[string]*types.Package // package id -> package object (just this package)
}

func (p *parser) init(filename, id string, src io.Reader, packages map[string]*types.Package) {
	p.scanner.Init(src)
	p.scanner.Error = func(_ *scanner.Scanner, msg string) { p.error(msg) }
	p.scanner.Mode = scanner.ScanIdents | scanner.ScanInts | scanner.ScanChars | scanner.ScanStrings | scanner.ScanComments | scanner.SkipComments
	p.scanner.Whitespace = 1<<'\t' | 1<<' '
	p.scanner.Filename = filename // for good error messages
	p.next()
	p.id = id
	p.sharedPkgs = packages
	if debug {
		// check consistency of packages map
		for _, pkg := range packages {
			if pkg.Name() == "" {
				fmt.Printf("no package name for %s\n", pkg.Path())
			}
		}
	}
}

func (p *parser) next() {
	p.tok = p.scanner.Scan()
	switch p.tok {
	case scanner.Ident, scanner.Int, scanner.Char, scanner.String, '·':
		p.lit = p.scanner.TokenText()
	default:
		p.lit = ""
	}
	if debug {
		fmt.Printf("%s: %q -> %q\n", scanner.TokenString(p.tok), p.scanner.TokenText(), p.lit)
	}
}

func declTypeName(pkg *types.Package, name string) *types.TypeName {
	scope := pkg.Scope()
	if obj := scope.Lookup(name); obj != nil {
		return obj.(*types.TypeName)
	}
	obj := types.NewTypeName(token.NoPos, pkg, name, nil)
	// a named type may be referred to before the underlying type
	// is known - set it up
	types.NewNamed(obj, nil, nil)
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

func (p *parser) error(err interface{}) {
	if s, ok := err.(string); ok {
		err = errors.New(s)
	}
	// panic with a runtime.Error if err is not an error
	panic(importError{p.scanner.Pos(), err.(error)})
}

func (p *parser) errorf(format string, args ...interface{}) {
	p.error(fmt.Sprintf(format, args...))
}

func (p *parser) expect(tok rune) string {
	lit := p.lit
	if p.tok != tok {
		p.errorf("expected %s, got %s (%s)", scanner.TokenString(tok), scanner.TokenString(p.tok), lit)
	}
	p.next()
	return lit
}

func (p *parser) expectSpecial(tok string) {
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

func (p *parser) expectKeyword(keyword string) {
	lit := p.expect(scanner.Ident)
	if lit != keyword {
		p.errorf("expected keyword %s, got %q", keyword, lit)
	}
}

// ----------------------------------------------------------------------------
// Qualified and unqualified names

// parsePackageID parses a PackageId:
//
//	PackageId = string_lit .
func (p *parser) parsePackageID() string {
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

// parsePackageName parse a PackageName:
//
//	PackageName = ident .
func (p *parser) parsePackageName() string {
	return p.expect(scanner.Ident)
}

// parseDotIdent parses a dotIdentifier:
//
//	dotIdentifier = ( ident | '·' ) { ident | int | '·' } .
func (p *parser) parseDotIdent() string {
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

// parseQualifiedName parses a QualifiedName:
//
//	QualifiedName = "@" PackageId "." ( "?" | dotIdentifier ) .
func (p *parser) parseQualifiedName() (id, name string) {
	p.expect('@')
	id = p.parsePackageID()
	p.expect('.')
	// Per rev f280b8a485fd (10/2/2013), qualified names may be used for anonymous fields.
	if p.tok == '?' {
		p.next()
	} else {
		name = p.parseDotIdent()
	}
	return
}

// getPkg returns the package for a given id. If the package is
// not found, create the package and add it to the p.localPkgs
// and p.sharedPkgs maps. name is the (expected) name of the
// package. If name == "", the package name is expected to be
// set later via an import clause in the export data.
//
// id identifies a package, usually by a canonical package path like
// "encoding/json" but possibly by a non-canonical import path like
// "./json".
func (p *parser) getPkg(id, name string) *types.Package {
	// package unsafe is not in the packages maps - handle explicitly
	if id == "unsafe" {
		return types.Unsafe
	}

	pkg := p.localPkgs[id]
	if pkg == nil {
		// first import of id from this package
		pkg = p.sharedPkgs[id]
		if pkg == nil {
			// first import of id by this importer;
			// add (possibly unnamed) pkg to shared packages
			pkg = types.NewPackage(id, name)
			p.sharedPkgs[id] = pkg
		}
		// add (possibly unnamed) pkg to local packages
		if p.localPkgs == nil {
			p.localPkgs = make(map[string]*types.Package)
		}
		p.localPkgs[id] = pkg
	} else if name != "" {
		// package exists already and we have an expected package name;
		// make sure names match or set package name if necessary
		if pname := pkg.Name(); pname == "" {
			pkg.SetName(name)
		} else if pname != name {
			p.errorf("%s package name mismatch: %s (given) vs %s (expected)", id, pname, name)
		}
	}
	return pkg
}

// parseExportedName is like parseQualifiedName, but
// the package id is resolved to an imported *types.Package.
func (p *parser) parseExportedName() (pkg *types.Package, name string) {
	id, name := p.parseQualifiedName()
	pkg = p.getPkg(id, "")
	return
}

// ----------------------------------------------------------------------------
// Types

// parseBasicType parses a BasicType:
//
//	BasicType = identifier .
func (p *parser) parseBasicType() types.Type {
	id := p.expect(scanner.Ident)
	obj := types.Universe.Lookup(id)
	if obj, ok := obj.(*types.TypeName); ok {
		return obj.Type()
	}
	p.errorf("not a basic type: %s", id)
	return nil
}

// parseArrayType parses an ArrayType:
//
//	ArrayType = "[" int_lit "]" Type .
func (p *parser) parseArrayType(parent *types.Package) types.Type {
	// "[" already consumed and lookahead known not to be "]"
	lit := p.expect(scanner.Int)
	p.expect(']')
	elem := p.parseType(parent)
	n, err := strconv.ParseInt(lit, 10, 64)
	if err != nil {
		p.error(err)
	}
	return types.NewArray(elem, n)
}

// parseMapType parses a MapType:
//
//	MapType = "map" "[" Type "]" Type .
func (p *parser) parseMapType(parent *types.Package) types.Type {
	p.expectKeyword("map")
	p.expect('[')
	key := p.parseType(parent)
	p.expect(']')
	elem := p.parseType(parent)
	return types.NewMap(key, elem)
}

// parseName parses a Name:
//
//	Name = identifier | "?" | QualifiedName .
//
// For unqualified and anonymous names, the returned package is the parent
// package unless parent == nil, in which case the returned package is the
// package being imported. (The parent package is not nil if the name
// is an unqualified struct field or interface method name belonging to a
// type declared in another package.)
//
// For qualified names, the returned package is nil (and not created if
// it doesn't exist yet) unless materializePkg is set (which creates an
// unnamed package with valid package path). In the latter case, a
// subsequent import clause is expected to provide a name for the package.
func (p *parser) parseName(parent *types.Package, materializePkg bool) (pkg *types.Package, name string) {
	pkg = parent
	if pkg == nil {
		pkg = p.sharedPkgs[p.id]
	}
	switch p.tok {
	case scanner.Ident:
		name = p.lit
		p.next()
	case '?':
		// anonymous
		p.next()
	case '@':
		// exported name prefixed with package path
		pkg = nil
		var id string
		id, name = p.parseQualifiedName()
		if materializePkg {
			pkg = p.getPkg(id, "")
		}
	default:
		p.error("name expected")
	}
	return
}

func deref(typ types.Type) types.Type {
	if p, _ := typ.(*types.Pointer); p != nil {
		return p.Elem()
	}
	return typ
}

// parseField parses a Field:
//
//	Field = Name Type [ string_lit ] .
func (p *parser) parseField(parent *types.Package) (*types.Var, string) {
	pkg, name := p.parseName(parent, true)

	if name == "_" {
		// Blank fields should be package-qualified because they
		// are unexported identifiers, but gc does not qualify them.
		// Assuming that the ident belongs to the current package
		// causes types to change during re-exporting, leading
		// to spurious "can't assign A to B" errors from go/types.
		// As a workaround, pretend all blank fields belong
		// to the same unique dummy package.
		const blankpkg = "<_>"
		pkg = p.getPkg(blankpkg, blankpkg)
	}

	typ := p.parseType(parent)
	anonymous := false
	if name == "" {
		// anonymous field - typ must be T or *T and T must be a type name
		switch typ := deref(typ).(type) {
		case *types.Basic: // basic types are named types
			pkg = nil // objects defined in Universe scope have no package
			name = typ.Name()
		case *types.Named:
			name = typ.Obj().Name()
		default:
			p.errorf("anonymous field expected")
		}
		anonymous = true
	}
	tag := ""
	if p.tok == scanner.String {
		s := p.expect(scanner.String)
		var err error
		tag, err = strconv.Unquote(s)
		if err != nil {
			p.errorf("invalid struct tag %s: %s", s, err)
		}
	}
	return types.NewField(token.NoPos, pkg, name, typ, anonymous), tag
}

// parseStructType parses a StructType:
//
//	StructType = "struct" "{" [ FieldList ] "}" .
//	FieldList  = Field { ";" Field } .
func (p *parser) parseStructType(parent *types.Package) types.Type {
	var fields []*types.Var
	var tags []string

	p.expectKeyword("struct")
	p.expect('{')
	for i := 0; p.tok != '}' && p.tok != scanner.EOF; i++ {
		if i > 0 {
			p.expect(';')
		}
		fld, tag := p.parseField(parent)
		if tag != "" && tags == nil {
			tags = make([]string, i)
		}
		if tags != nil {
			tags = append(tags, tag)
		}
		fields = append(fields, fld)
	}
	p.expect('}')

	return types.NewStruct(fields, tags)
}

// parseParameter parses a Parameter:
//
//	Parameter = ( identifier | "?" ) [ "..." ] Type [ string_lit ] .
func (p *parser) parseParameter() (par *types.Var, isVariadic bool) {
	_, name := p.parseName(nil, false)
	// remove gc-specific parameter numbering
	if i := strings.Index(name, "·"); i >= 0 {
		name = name[:i]
	}
	if p.tok == '.' {
		p.expectSpecial("...")
		isVariadic = true
	}
	typ := p.parseType(nil)
	if isVariadic {
		typ = types.NewSlice(typ)
	}
	// ignore argument tag (e.g. "noescape")
	if p.tok == scanner.String {
		p.next()
	}
	// TODO(gri) should we provide a package?
	par = types.NewVar(token.NoPos, nil, name, typ)
	return
}

// parseParameters parses a Parameters:
//
//	Parameters    = "(" [ ParameterList ] ")" .
//	ParameterList = { Parameter "," } Parameter .
func (p *parser) parseParameters() (list []*types.Var, isVariadic bool) {
	p.expect('(')
	for p.tok != ')' && p.tok != scanner.EOF {
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

// parseSignature parses a Signature:
//
//	Signature = Parameters [ Result ] .
//	Result    = Type | Parameters .
func (p *parser) parseSignature(recv *types.Var) *types.Signature {
	params, isVariadic := p.parseParameters()

	// optional result type
	var results []*types.Var
	if p.tok == '(' {
		var variadic bool
		results, variadic = p.parseParameters()
		if variadic {
			p.error("... not permitted on result type")
		}
	}

	return types.NewSignature(recv, types.NewTuple(params...), types.NewTuple(results...), isVariadic)
}

// parseInterfaceType parses an InterfaceType:
//
//	InterfaceType = "interface" "{" [ MethodList ] "}" .
//	MethodList    = Method { ";" Method } .
//	Method        = Name Signature .
//
// The methods of embedded interfaces are always "inlined"
// by the compiler and thus embedded interfaces are never
// visible in the export data.
func (p *parser) parseInterfaceType(parent *types.Package) types.Type {
	var methods []*types.Func

	p.expectKeyword("interface")
	p.expect('{')
	for i := 0; p.tok != '}' && p.tok != scanner.EOF; i++ {
		if i > 0 {
			p.expect(';')
		}
		pkg, name := p.parseName(parent, true)
		sig := p.parseSignature(nil)
		methods = append(methods, types.NewFunc(token.NoPos, pkg, name, sig))
	}
	p.expect('}')

	// Complete requires the type's embedded interfaces to be fully defined,
	// but we do not define any
	return newInterface(methods, nil).Complete()
}

// parseChanType parses a ChanType:
//
//	ChanType = ( "chan" [ "<-" ] | "<-" "chan" ) Type .
func (p *parser) parseChanType(parent *types.Package) types.Type {
	dir := types.SendRecv
	if p.tok == scanner.Ident {
		p.expectKeyword("chan")
		if p.tok == '<' {
			p.expectSpecial("<-")
			dir = types.SendOnly
		}
	} else {
		p.expectSpecial("<-")
		p.expectKeyword("chan")
		dir = types.RecvOnly
	}
	elem := p.parseType(parent)
	return types.NewChan(dir, elem)
}

// parseType parses a Type:
//
//	Type =
//		BasicType | TypeName | ArrayType | SliceType | StructType |
//		PointerType | FuncType | InterfaceType | MapType | ChanType |
//		"(" Type ")" .
//
//	BasicType   = ident .
//	TypeName    = ExportedName .
//	SliceType   = "[" "]" Type .
//	PointerType = "*" Type .
//	FuncType    = "func" Signature .
func (p *parser) parseType(parent *types.Package) types.Type {
	switch p.tok {
	case scanner.Ident:
		switch p.lit {
		default:
			return p.parseBasicType()
		case "struct":
			return p.parseStructType(parent)
		case "func":
			// FuncType
			p.next()
			return p.parseSignature(nil)
		case "interface":
			return p.parseInterfaceType(parent)
		case "map":
			return p.parseMapType(parent)
		case "chan":
			return p.parseChanType(parent)
		}
	case '@':
		// TypeName
		pkg, name := p.parseExportedName()
		return declTypeName(pkg, name).Type()
	case '[':
		p.next() // look ahead
		if p.tok == ']' {
			// SliceType
			p.next()
			return types.NewSlice(p.parseType(parent))
		}
		return p.parseArrayType(parent)
	case '*':
		// PointerType
		p.next()
		return types.NewPointer(p.parseType(parent))
	case '<':
		return p.parseChanType(parent)
	case '(':
		// "(" Type ")"
		p.next()
		typ := p.parseType(parent)
		p.expect(')')
		return typ
	}
	p.errorf("expected type, got %s (%q)", scanner.TokenString(p.tok), p.lit)
	return nil
}

// ----------------------------------------------------------------------------
// Declarations

// parseImportDecl parses an ImportDecl:
//
//	ImportDecl = "import" PackageName PackageId .
func (p *parser) parseImportDecl() {
	p.expectKeyword("import")
	name := p.parsePackageName()
	p.getPkg(p.parsePackageID(), name)
}

// parseInt parses an int_lit:
//
//	int_lit = [ "+" | "-" ] { "0" ... "9" } .
func (p *parser) parseInt() string {
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

// parseNumber parses a number:
//
//	number = int_lit [ "p" int_lit ] .
func (p *parser) parseNumber() (typ *types.Basic, val constant.Value) {
	// mantissa
	mant := constant.MakeFromLiteral(p.parseInt(), token.INT, 0)
	if mant == nil {
		panic("invalid mantissa")
	}

	if p.lit == "p" {
		// exponent (base 2)
		p.next()
		exp, err := strconv.ParseInt(p.parseInt(), 10, 0)
		if err != nil {
			p.error(err)
		}
		if exp < 0 {
			denom := constant.MakeInt64(1)
			denom = constant.Shift(denom, token.SHL, uint(-exp))
			typ = types.Typ[types.UntypedFloat]
			val = constant.BinaryOp(mant, token.QUO, denom)
			return
		}
		if exp > 0 {
			mant = constant.Shift(mant, token.SHL, uint(exp))
		}
		typ = types.Typ[types.UntypedFloat]
		val = mant
		return
	}

	typ = types.Typ[types.UntypedInt]
	val = mant
	return
}

// parseConstDecl parses a ConstDecl:
//
//	ConstDecl   = "const" ExportedName [ Type ] "=" Literal .
//	Literal     = bool_lit | int_lit | float_lit | complex_lit | rune_lit | string_lit .
//	bool_lit    = "true" | "false" .
//	complex_lit = "(" float_lit "+" float_lit "i" ")" .
//	rune_lit    = "(" int_lit "+" int_lit ")" .
//	string_lit  = `"` { unicode_char } `"` .
func (p *parser) parseConstDecl() {
	p.expectKeyword("const")
	pkg, name := p.parseExportedName()

	var typ0 types.Type
	if p.tok != '=' {
		// constant types are never structured - no need for parent type
		typ0 = p.parseType(nil)
	}

	p.expect('=')
	var typ types.Type
	var val constant.Value
	switch p.tok {
	case scanner.Ident:
		// bool_lit
		if p.lit != "true" && p.lit != "false" {
			p.error("expected true or false")
		}
		typ = types.Typ[types.UntypedBool]
		val = constant.MakeBool(p.lit == "true")
		p.next()

	case '-', scanner.Int:
		// int_lit
		typ, val = p.parseNumber()

	case '(':
		// complex_lit or rune_lit
		p.next()
		if p.tok == scanner.Char {
			p.next()
			p.expect('+')
			typ = types.Typ[types.UntypedRune]
			_, val = p.parseNumber()
			p.expect(')')
			break
		}
		_, re := p.parseNumber()
		p.expect('+')
		_, im := p.parseNumber()
		p.expectKeyword("i")
		p.expect(')')
		typ = types.Typ[types.UntypedComplex]
		val = constant.BinaryOp(re, token.ADD, constant.MakeImag(im))

	case scanner.Char:
		// rune_lit
		typ = types.Typ[types.UntypedRune]
		val = constant.MakeFromLiteral(p.lit, token.CHAR, 0)
		p.next()

	case scanner.String:
		// string_lit
		typ = types.Typ[types.UntypedString]
		val = constant.MakeFromLiteral(p.lit, token.STRING, 0)
		p.next()

	default:
		p.errorf("expected literal got %s", scanner.TokenString(p.tok))
	}

	if typ0 == nil {
		typ0 = typ
	}

	pkg.Scope().Insert(types.NewConst(token.NoPos, pkg, name, typ0, val))
}

// parseTypeDecl parses a TypeDecl:
//
//	TypeDecl = "type" ExportedName Type .
func (p *parser) parseTypeDecl() {
	p.expectKeyword("type")
	pkg, name := p.parseExportedName()
	obj := declTypeName(pkg, name)

	// The type object may have been imported before and thus already
	// have a type associated with it. We still need to parse the type
	// structure, but throw it away if the object already has a type.
	// This ensures that all imports refer to the same type object for
	// a given type declaration.
	typ := p.parseType(pkg)

	if name := obj.Type().(*types.Named); name.Underlying() == nil {
		name.SetUnderlying(typ)
	}
}

// parseVarDecl parses a VarDecl:
//
//	VarDecl = "var" ExportedName Type .
func (p *parser) parseVarDecl() {
	p.expectKeyword("var")
	pkg, name := p.parseExportedName()
	typ := p.parseType(pkg)
	pkg.Scope().Insert(types.NewVar(token.NoPos, pkg, name, typ))
}

// parseFunc parses a Func:
//
//	Func = Signature [ Body ] .
//	Body = "{" ... "}" .
func (p *parser) parseFunc(recv *types.Var) *types.Signature {
	sig := p.parseSignature(recv)
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

// parseMethodDecl parses a MethodDecl:
//
//	MethodDecl = "func" Receiver Name Func .
//	Receiver   = "(" ( identifier | "?" ) [ "*" ] ExportedName ")" .
func (p *parser) parseMethodDecl() {
	// "func" already consumed
	p.expect('(')
	recv, _ := p.parseParameter() // receiver
	p.expect(')')

	// determine receiver base type object
	base := deref(recv.Type()).(*types.Named)

	// parse method name, signature, and possibly inlined body
	_, name := p.parseName(nil, false)
	sig := p.parseFunc(recv)

	// methods always belong to the same package as the base type object
	pkg := base.Obj().Pkg()

	// add method to type unless type was imported before
	// and method exists already
	// TODO(gri) This leads to a quadratic algorithm - ok for now because method counts are small.
	base.AddMethod(types.NewFunc(token.NoPos, pkg, name, sig))
}

// parseFuncDecl parses a FuncDecl:
//
//	FuncDecl = "func" ExportedName Func .
func (p *parser) parseFuncDecl() {
	// "func" already consumed
	pkg, name := p.parseExportedName()
	typ := p.parseFunc(nil)
	pkg.Scope().Insert(types.NewFunc(token.NoPos, pkg, name, typ))
}

// parseDecl parses a Decl:
//
//	Decl = [ ImportDecl | ConstDecl | TypeDecl | VarDecl | FuncDecl | MethodDecl ] "\n" .
func (p *parser) parseDecl() {
	if p.tok == scanner.Ident {
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
	}
	p.expect('\n')
}

// ----------------------------------------------------------------------------
// Export

// parseExport parses an Export:
//
//	Export        = "PackageClause { Decl } "$$" .
//	PackageClause = "package" PackageName [ "safe" ] "\n" .
func (p *parser) parseExport() *types.Package {
	p.expectKeyword("package")
	name := p.parsePackageName()
	if p.tok == scanner.Ident && p.lit == "safe" {
		// package was compiled with -u option - ignore
		p.next()
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

	// Record all locally referenced packages as imports.
	var imports []*types.Package
	for id, pkg2 := range p.localPkgs {
		if pkg2.Name() == "" {
			p.errorf("%s package has no name", id)
		}
		if id == p.id {
			continue // avoid self-edge
		}
		imports = append(imports, pkg2)
	}
	sort.Sort(byPath(imports))
	pkg.SetImports(imports)

	// package was imported completely and without errors
	pkg.MarkComplete()

	return pkg
}

type byPath []*types.Package

func (a byPath) Len() int           { return len(a) }
func (a byPath) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a byPath) Less(i, j int) bool { return a[i].Path() < a[j].Path() }

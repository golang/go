// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Binary package export.
// This file was derived from $GOROOT/src/cmd/compile/internal/gc/bexport.go;
// see that file for specification of the format.

package gcimporter

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"go/ast"
	"go/constant"
	"go/token"
	"go/types"
	"math"
	"math/big"
	"sort"
	"strings"
)

// If debugFormat is set, each integer and string value is preceded by a marker
// and position information in the encoding. This mechanism permits an importer
// to recognize immediately when it is out of sync. The importer recognizes this
// mode automatically (i.e., it can import export data produced with debugging
// support even if debugFormat is not set at the time of import). This mode will
// lead to massively larger export data (by a factor of 2 to 3) and should only
// be enabled during development and debugging.
//
// NOTE: This flag is the first flag to enable if importing dies because of
// (suspected) format errors, and whenever a change is made to the format.
const debugFormat = false // default: false

// If trace is set, debugging output is printed to std out.
const trace = false // default: false

// Current export format version. Increase with each format change.
// Note: The latest binary (non-indexed) export format is at version 6.
//       This exporter is still at level 4, but it doesn't matter since
//       the binary importer can handle older versions just fine.
// 6: package height (CL 105038) -- NOT IMPLEMENTED HERE
// 5: improved position encoding efficiency (issue 20080, CL 41619) -- NOT IMPLEMEMTED HERE
// 4: type name objects support type aliases, uses aliasTag
// 3: Go1.8 encoding (same as version 2, aliasTag defined but never used)
// 2: removed unused bool in ODCL export (compiler only)
// 1: header format change (more regular), export package for _ struct fields
// 0: Go1.7 encoding
const exportVersion = 4

// trackAllTypes enables cycle tracking for all types, not just named
// types. The existing compiler invariants assume that unnamed types
// that are not completely set up are not used, or else there are spurious
// errors.
// If disabled, only named types are tracked, possibly leading to slightly
// less efficient encoding in rare cases. It also prevents the export of
// some corner-case type declarations (but those are not handled correctly
// with with the textual export format either).
// TODO(gri) enable and remove once issues caused by it are fixed
const trackAllTypes = false

type exporter struct {
	fset *token.FileSet
	out  bytes.Buffer

	// object -> index maps, indexed in order of serialization
	strIndex map[string]int
	pkgIndex map[*types.Package]int
	typIndex map[types.Type]int

	// position encoding
	posInfoFormat bool
	prevFile      string
	prevLine      int

	// debugging support
	written int // bytes written
	indent  int // for trace
}

// internalError represents an error generated inside this package.
type internalError string

func (e internalError) Error() string { return "gcimporter: " + string(e) }

func internalErrorf(format string, args ...interface{}) error {
	return internalError(fmt.Sprintf(format, args...))
}

// BExportData returns binary export data for pkg.
// If no file set is provided, position info will be missing.
func BExportData(fset *token.FileSet, pkg *types.Package) (b []byte, err error) {
	defer func() {
		if e := recover(); e != nil {
			if ierr, ok := e.(internalError); ok {
				err = ierr
				return
			}
			// Not an internal error; panic again.
			panic(e)
		}
	}()

	p := exporter{
		fset:          fset,
		strIndex:      map[string]int{"": 0}, // empty string is mapped to 0
		pkgIndex:      make(map[*types.Package]int),
		typIndex:      make(map[types.Type]int),
		posInfoFormat: true, // TODO(gri) might become a flag, eventually
	}

	// write version info
	// The version string must start with "version %d" where %d is the version
	// number. Additional debugging information may follow after a blank; that
	// text is ignored by the importer.
	p.rawStringln(fmt.Sprintf("version %d", exportVersion))
	var debug string
	if debugFormat {
		debug = "debug"
	}
	p.rawStringln(debug) // cannot use p.bool since it's affected by debugFormat; also want to see this clearly
	p.bool(trackAllTypes)
	p.bool(p.posInfoFormat)

	// --- generic export data ---

	// populate type map with predeclared "known" types
	for index, typ := range predeclared {
		p.typIndex[typ] = index
	}
	if len(p.typIndex) != len(predeclared) {
		return nil, internalError("duplicate entries in type map?")
	}

	// write package data
	p.pkg(pkg, true)
	if trace {
		p.tracef("\n")
	}

	// write objects
	objcount := 0
	scope := pkg.Scope()
	for _, name := range scope.Names() {
		if !ast.IsExported(name) {
			continue
		}
		if trace {
			p.tracef("\n")
		}
		p.obj(scope.Lookup(name))
		objcount++
	}

	// indicate end of list
	if trace {
		p.tracef("\n")
	}
	p.tag(endTag)

	// for self-verification only (redundant)
	p.int(objcount)

	if trace {
		p.tracef("\n")
	}

	// --- end of export data ---

	return p.out.Bytes(), nil
}

func (p *exporter) pkg(pkg *types.Package, emptypath bool) {
	if pkg == nil {
		panic(internalError("unexpected nil pkg"))
	}

	// if we saw the package before, write its index (>= 0)
	if i, ok := p.pkgIndex[pkg]; ok {
		p.index('P', i)
		return
	}

	// otherwise, remember the package, write the package tag (< 0) and package data
	if trace {
		p.tracef("P%d = { ", len(p.pkgIndex))
		defer p.tracef("} ")
	}
	p.pkgIndex[pkg] = len(p.pkgIndex)

	p.tag(packageTag)
	p.string(pkg.Name())
	if emptypath {
		p.string("")
	} else {
		p.string(pkg.Path())
	}
}

func (p *exporter) obj(obj types.Object) {
	switch obj := obj.(type) {
	case *types.Const:
		p.tag(constTag)
		p.pos(obj)
		p.qualifiedName(obj)
		p.typ(obj.Type())
		p.value(obj.Val())

	case *types.TypeName:
		if obj.IsAlias() {
			p.tag(aliasTag)
			p.pos(obj)
			p.qualifiedName(obj)
		} else {
			p.tag(typeTag)
		}
		p.typ(obj.Type())

	case *types.Var:
		p.tag(varTag)
		p.pos(obj)
		p.qualifiedName(obj)
		p.typ(obj.Type())

	case *types.Func:
		p.tag(funcTag)
		p.pos(obj)
		p.qualifiedName(obj)
		sig := obj.Type().(*types.Signature)
		p.paramList(sig.Params(), sig.Variadic())
		p.paramList(sig.Results(), false)

	default:
		panic(internalErrorf("unexpected object %v (%T)", obj, obj))
	}
}

func (p *exporter) pos(obj types.Object) {
	if !p.posInfoFormat {
		return
	}

	file, line := p.fileLine(obj)
	if file == p.prevFile {
		// common case: write line delta
		// delta == 0 means different file or no line change
		delta := line - p.prevLine
		p.int(delta)
		if delta == 0 {
			p.int(-1) // -1 means no file change
		}
	} else {
		// different file
		p.int(0)
		// Encode filename as length of common prefix with previous
		// filename, followed by (possibly empty) suffix. Filenames
		// frequently share path prefixes, so this can save a lot
		// of space and make export data size less dependent on file
		// path length. The suffix is unlikely to be empty because
		// file names tend to end in ".go".
		n := commonPrefixLen(p.prevFile, file)
		p.int(n)           // n >= 0
		p.string(file[n:]) // write suffix only
		p.prevFile = file
		p.int(line)
	}
	p.prevLine = line
}

func (p *exporter) fileLine(obj types.Object) (file string, line int) {
	if p.fset != nil {
		pos := p.fset.Position(obj.Pos())
		file = pos.Filename
		line = pos.Line
	}
	return
}

func commonPrefixLen(a, b string) int {
	if len(a) > len(b) {
		a, b = b, a
	}
	// len(a) <= len(b)
	i := 0
	for i < len(a) && a[i] == b[i] {
		i++
	}
	return i
}

func (p *exporter) qualifiedName(obj types.Object) {
	p.string(obj.Name())
	p.pkg(obj.Pkg(), false)
}

func (p *exporter) typ(t types.Type) {
	if t == nil {
		panic(internalError("nil type"))
	}

	// Possible optimization: Anonymous pointer types *T where
	// T is a named type are common. We could canonicalize all
	// such types *T to a single type PT = *T. This would lead
	// to at most one *T entry in typIndex, and all future *T's
	// would be encoded as the respective index directly. Would
	// save 1 byte (pointerTag) per *T and reduce the typIndex
	// size (at the cost of a canonicalization map). We can do
	// this later, without encoding format change.

	// if we saw the type before, write its index (>= 0)
	if i, ok := p.typIndex[t]; ok {
		p.index('T', i)
		return
	}

	// otherwise, remember the type, write the type tag (< 0) and type data
	if trackAllTypes {
		if trace {
			p.tracef("T%d = {>\n", len(p.typIndex))
			defer p.tracef("<\n} ")
		}
		p.typIndex[t] = len(p.typIndex)
	}

	switch t := t.(type) {
	case *types.Named:
		if !trackAllTypes {
			// if we don't track all types, track named types now
			p.typIndex[t] = len(p.typIndex)
		}

		p.tag(namedTag)
		p.pos(t.Obj())
		p.qualifiedName(t.Obj())
		p.typ(t.Underlying())
		if !types.IsInterface(t) {
			p.assocMethods(t)
		}

	case *types.Array:
		p.tag(arrayTag)
		p.int64(t.Len())
		p.typ(t.Elem())

	case *types.Slice:
		p.tag(sliceTag)
		p.typ(t.Elem())

	case *dddSlice:
		p.tag(dddTag)
		p.typ(t.elem)

	case *types.Struct:
		p.tag(structTag)
		p.fieldList(t)

	case *types.Pointer:
		p.tag(pointerTag)
		p.typ(t.Elem())

	case *types.Signature:
		p.tag(signatureTag)
		p.paramList(t.Params(), t.Variadic())
		p.paramList(t.Results(), false)

	case *types.Interface:
		p.tag(interfaceTag)
		p.iface(t)

	case *types.Map:
		p.tag(mapTag)
		p.typ(t.Key())
		p.typ(t.Elem())

	case *types.Chan:
		p.tag(chanTag)
		p.int(int(3 - t.Dir())) // hack
		p.typ(t.Elem())

	default:
		panic(internalErrorf("unexpected type %T: %s", t, t))
	}
}

func (p *exporter) assocMethods(named *types.Named) {
	// Sort methods (for determinism).
	var methods []*types.Func
	for i := 0; i < named.NumMethods(); i++ {
		methods = append(methods, named.Method(i))
	}
	sort.Sort(methodsByName(methods))

	p.int(len(methods))

	if trace && methods != nil {
		p.tracef("associated methods {>\n")
	}

	for i, m := range methods {
		if trace && i > 0 {
			p.tracef("\n")
		}

		p.pos(m)
		name := m.Name()
		p.string(name)
		if !exported(name) {
			p.pkg(m.Pkg(), false)
		}

		sig := m.Type().(*types.Signature)
		p.paramList(types.NewTuple(sig.Recv()), false)
		p.paramList(sig.Params(), sig.Variadic())
		p.paramList(sig.Results(), false)
		p.int(0) // dummy value for go:nointerface pragma - ignored by importer
	}

	if trace && methods != nil {
		p.tracef("<\n} ")
	}
}

type methodsByName []*types.Func

func (x methodsByName) Len() int           { return len(x) }
func (x methodsByName) Swap(i, j int)      { x[i], x[j] = x[j], x[i] }
func (x methodsByName) Less(i, j int) bool { return x[i].Name() < x[j].Name() }

func (p *exporter) fieldList(t *types.Struct) {
	if trace && t.NumFields() > 0 {
		p.tracef("fields {>\n")
		defer p.tracef("<\n} ")
	}

	p.int(t.NumFields())
	for i := 0; i < t.NumFields(); i++ {
		if trace && i > 0 {
			p.tracef("\n")
		}
		p.field(t.Field(i))
		p.string(t.Tag(i))
	}
}

func (p *exporter) field(f *types.Var) {
	if !f.IsField() {
		panic(internalError("field expected"))
	}

	p.pos(f)
	p.fieldName(f)
	p.typ(f.Type())
}

func (p *exporter) iface(t *types.Interface) {
	// TODO(gri): enable importer to load embedded interfaces,
	// then emit Embeddeds and ExplicitMethods separately here.
	p.int(0)

	n := t.NumMethods()
	if trace && n > 0 {
		p.tracef("methods {>\n")
		defer p.tracef("<\n} ")
	}
	p.int(n)
	for i := 0; i < n; i++ {
		if trace && i > 0 {
			p.tracef("\n")
		}
		p.method(t.Method(i))
	}
}

func (p *exporter) method(m *types.Func) {
	sig := m.Type().(*types.Signature)
	if sig.Recv() == nil {
		panic(internalError("method expected"))
	}

	p.pos(m)
	p.string(m.Name())
	if m.Name() != "_" && !ast.IsExported(m.Name()) {
		p.pkg(m.Pkg(), false)
	}

	// interface method; no need to encode receiver.
	p.paramList(sig.Params(), sig.Variadic())
	p.paramList(sig.Results(), false)
}

func (p *exporter) fieldName(f *types.Var) {
	name := f.Name()

	if f.Anonymous() {
		// anonymous field - we distinguish between 3 cases:
		// 1) field name matches base type name and is exported
		// 2) field name matches base type name and is not exported
		// 3) field name doesn't match base type name (alias name)
		bname := basetypeName(f.Type())
		if name == bname {
			if ast.IsExported(name) {
				name = "" // 1) we don't need to know the field name or package
			} else {
				name = "?" // 2) use unexported name "?" to force package export
			}
		} else {
			// 3) indicate alias and export name as is
			// (this requires an extra "@" but this is a rare case)
			p.string("@")
		}
	}

	p.string(name)
	if name != "" && !ast.IsExported(name) {
		p.pkg(f.Pkg(), false)
	}
}

func basetypeName(typ types.Type) string {
	switch typ := deref(typ).(type) {
	case *types.Basic:
		return typ.Name()
	case *types.Named:
		return typ.Obj().Name()
	default:
		return "" // unnamed type
	}
}

func (p *exporter) paramList(params *types.Tuple, variadic bool) {
	// use negative length to indicate unnamed parameters
	// (look at the first parameter only since either all
	// names are present or all are absent)
	n := params.Len()
	if n > 0 && params.At(0).Name() == "" {
		n = -n
	}
	p.int(n)
	for i := 0; i < params.Len(); i++ {
		q := params.At(i)
		t := q.Type()
		if variadic && i == params.Len()-1 {
			t = &dddSlice{t.(*types.Slice).Elem()}
		}
		p.typ(t)
		if n > 0 {
			name := q.Name()
			p.string(name)
			if name != "_" {
				p.pkg(q.Pkg(), false)
			}
		}
		p.string("") // no compiler-specific info
	}
}

func (p *exporter) value(x constant.Value) {
	if trace {
		p.tracef("= ")
	}

	switch x.Kind() {
	case constant.Bool:
		tag := falseTag
		if constant.BoolVal(x) {
			tag = trueTag
		}
		p.tag(tag)

	case constant.Int:
		if v, exact := constant.Int64Val(x); exact {
			// common case: x fits into an int64 - use compact encoding
			p.tag(int64Tag)
			p.int64(v)
			return
		}
		// uncommon case: large x - use float encoding
		// (powers of 2 will be encoded efficiently with exponent)
		p.tag(floatTag)
		p.float(constant.ToFloat(x))

	case constant.Float:
		p.tag(floatTag)
		p.float(x)

	case constant.Complex:
		p.tag(complexTag)
		p.float(constant.Real(x))
		p.float(constant.Imag(x))

	case constant.String:
		p.tag(stringTag)
		p.string(constant.StringVal(x))

	case constant.Unknown:
		// package contains type errors
		p.tag(unknownTag)

	default:
		panic(internalErrorf("unexpected value %v (%T)", x, x))
	}
}

func (p *exporter) float(x constant.Value) {
	if x.Kind() != constant.Float {
		panic(internalErrorf("unexpected constant %v, want float", x))
	}
	// extract sign (there is no -0)
	sign := constant.Sign(x)
	if sign == 0 {
		// x == 0
		p.int(0)
		return
	}
	// x != 0

	var f big.Float
	if v, exact := constant.Float64Val(x); exact {
		// float64
		f.SetFloat64(v)
	} else if num, denom := constant.Num(x), constant.Denom(x); num.Kind() == constant.Int {
		// TODO(gri): add big.Rat accessor to constant.Value.
		r := valueToRat(num)
		f.SetRat(r.Quo(r, valueToRat(denom)))
	} else {
		// Value too large to represent as a fraction => inaccessible.
		// TODO(gri): add big.Float accessor to constant.Value.
		f.SetFloat64(math.MaxFloat64) // FIXME
	}

	// extract exponent such that 0.5 <= m < 1.0
	var m big.Float
	exp := f.MantExp(&m)

	// extract mantissa as *big.Int
	// - set exponent large enough so mant satisfies mant.IsInt()
	// - get *big.Int from mant
	m.SetMantExp(&m, int(m.MinPrec()))
	mant, acc := m.Int(nil)
	if acc != big.Exact {
		panic(internalError("internal error"))
	}

	p.int(sign)
	p.int(exp)
	p.string(string(mant.Bytes()))
}

func valueToRat(x constant.Value) *big.Rat {
	// Convert little-endian to big-endian.
	// I can't believe this is necessary.
	bytes := constant.Bytes(x)
	for i := 0; i < len(bytes)/2; i++ {
		bytes[i], bytes[len(bytes)-1-i] = bytes[len(bytes)-1-i], bytes[i]
	}
	return new(big.Rat).SetInt(new(big.Int).SetBytes(bytes))
}

func (p *exporter) bool(b bool) bool {
	if trace {
		p.tracef("[")
		defer p.tracef("= %v] ", b)
	}

	x := 0
	if b {
		x = 1
	}
	p.int(x)
	return b
}

// ----------------------------------------------------------------------------
// Low-level encoders

func (p *exporter) index(marker byte, index int) {
	if index < 0 {
		panic(internalError("invalid index < 0"))
	}
	if debugFormat {
		p.marker('t')
	}
	if trace {
		p.tracef("%c%d ", marker, index)
	}
	p.rawInt64(int64(index))
}

func (p *exporter) tag(tag int) {
	if tag >= 0 {
		panic(internalError("invalid tag >= 0"))
	}
	if debugFormat {
		p.marker('t')
	}
	if trace {
		p.tracef("%s ", tagString[-tag])
	}
	p.rawInt64(int64(tag))
}

func (p *exporter) int(x int) {
	p.int64(int64(x))
}

func (p *exporter) int64(x int64) {
	if debugFormat {
		p.marker('i')
	}
	if trace {
		p.tracef("%d ", x)
	}
	p.rawInt64(x)
}

func (p *exporter) string(s string) {
	if debugFormat {
		p.marker('s')
	}
	if trace {
		p.tracef("%q ", s)
	}
	// if we saw the string before, write its index (>= 0)
	// (the empty string is mapped to 0)
	if i, ok := p.strIndex[s]; ok {
		p.rawInt64(int64(i))
		return
	}
	// otherwise, remember string and write its negative length and bytes
	p.strIndex[s] = len(p.strIndex)
	p.rawInt64(-int64(len(s)))
	for i := 0; i < len(s); i++ {
		p.rawByte(s[i])
	}
}

// marker emits a marker byte and position information which makes
// it easy for a reader to detect if it is "out of sync". Used for
// debugFormat format only.
func (p *exporter) marker(m byte) {
	p.rawByte(m)
	// Enable this for help tracking down the location
	// of an incorrect marker when running in debugFormat.
	if false && trace {
		p.tracef("#%d ", p.written)
	}
	p.rawInt64(int64(p.written))
}

// rawInt64 should only be used by low-level encoders.
func (p *exporter) rawInt64(x int64) {
	var tmp [binary.MaxVarintLen64]byte
	n := binary.PutVarint(tmp[:], x)
	for i := 0; i < n; i++ {
		p.rawByte(tmp[i])
	}
}

// rawStringln should only be used to emit the initial version string.
func (p *exporter) rawStringln(s string) {
	for i := 0; i < len(s); i++ {
		p.rawByte(s[i])
	}
	p.rawByte('\n')
}

// rawByte is the bottleneck interface to write to p.out.
// rawByte escapes b as follows (any encoding does that
// hides '$'):
//
//	'$'  => '|' 'S'
//	'|'  => '|' '|'
//
// Necessary so other tools can find the end of the
// export data by searching for "$$".
// rawByte should only be used by low-level encoders.
func (p *exporter) rawByte(b byte) {
	switch b {
	case '$':
		// write '$' as '|' 'S'
		b = 'S'
		fallthrough
	case '|':
		// write '|' as '|' '|'
		p.out.WriteByte('|')
		p.written++
	}
	p.out.WriteByte(b)
	p.written++
}

// tracef is like fmt.Printf but it rewrites the format string
// to take care of indentation.
func (p *exporter) tracef(format string, args ...interface{}) {
	if strings.ContainsAny(format, "<>\n") {
		var buf bytes.Buffer
		for i := 0; i < len(format); i++ {
			// no need to deal with runes
			ch := format[i]
			switch ch {
			case '>':
				p.indent++
				continue
			case '<':
				p.indent--
				continue
			}
			buf.WriteByte(ch)
			if ch == '\n' {
				for j := p.indent; j > 0; j-- {
					buf.WriteString(".  ")
				}
			}
		}
		format = buf.String()
	}
	fmt.Printf(format, args...)
}

// Debugging support.
// (tagString is only used when tracing is enabled)
var tagString = [...]string{
	// Packages
	-packageTag: "package",

	// Types
	-namedTag:     "named type",
	-arrayTag:     "array",
	-sliceTag:     "slice",
	-dddTag:       "ddd",
	-structTag:    "struct",
	-pointerTag:   "pointer",
	-signatureTag: "signature",
	-interfaceTag: "interface",
	-mapTag:       "map",
	-chanTag:      "chan",

	// Values
	-falseTag:    "false",
	-trueTag:     "true",
	-int64Tag:    "int64",
	-floatTag:    "float",
	-fractionTag: "fraction",
	-complexTag:  "complex",
	-stringTag:   "string",
	-unknownTag:  "unknown",

	// Type aliases
	-aliasTag: "alias",
}

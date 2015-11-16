// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Binary package export.
// Based loosely on x/tools/go/importer.
// (see fmt.go, go.y as "documentation" for how to use/setup data structures)
//
// Use "-newexport" flag to enable.

// TODO(gri):
// - inlined functions

/*
Export data encoding:

The export data is a serialized description of the graph of exported
objects: constants, types, variables, and functions. Only types can
be re-exported and so we need to know which package they are coming
from. Therefore, packages are also part of the export graph.

The roots of the graph are the list of constants, variables, functions,
and eventually types. Types are written last because most of them will
be written as part of other objects which will reduce the number of
types that need to be written separately.

The graph is serialized in in-order fashion, starting with the roots.
Each object in the graph is serialized by writing its fields sequentially.
If the field is a pointer to another object, that object is serialized,
recursively. Otherwise the field is written. Non-pointer fields are all
encoded as either an integer or string value.

Only packages and types may be referred to more than once. When getting
to a package or type that was not serialized before, a number (index) is
assigned to it, starting at 0. In this case, the encoding starts with an
integer tag with a value < 0. The tag value indicates the kind of object
(package or type) that follows and that this is the first time that we
see this object. If the package or tag was already serialized, the encoding
starts with the respective package or type index >= 0. An importer can
trivially determine if a package or type needs to be read in for the first
time (tag < 0) and entered into the respective package or type table, or
if the package or type was seen already (index >= 0), in which case the
index is the table index where the respective object can be found.

Before exporting or importing, the type tables are populated with the
predeclared types (int, string, error, unsafe.Pointer, etc.). This way
they are automatically encoded with a known and fixed type index.

Encoding format:

The export data starts with a single byte indicating the encoding format
(compact, or with debugging information), followed by a version string
(so we can evolve the encoding if need be), the name of the imported
package, and a string containing platform-specific information for that
package.

After this header, the lists of objects follow. After the objects, platform-
specific data may be found which is not used strictly for type checking.

The encoding of objects is straight-forward: Constants, variables, and
functions start with their name, type, and possibly a value. Named types
record their name and package so that they can be canonicalized: If the
same type was imported before via another import, the importer must use
the previously imported type pointer so that we have exactly one version
(i.e., one pointer) for each named type (and read but discard the current
type encoding). Unnamed types simply encode their respective fields.

In the encoding, all lists (of objects, struct fields, methods, parameter
names, but also the bytes of a string, etc.) start with an integer which
is the length of the list. This permits an importer to allocate the right
amount of space to hold the list without the need to grow it later.

All integer values use a variable-length encoding for compact representation.

If debugFormat is set, each integer and string value is preceeded by a marker
and position information in the encoding. This mechanism permits an importer
to recognize immediately when it is out of sync. The importer recognizes this
mode automatically (i.e., it can import export data produced with debugging
support even if debugFormat is not set at the time of import). Using this mode
will massively increase the size of the export data (by a factor of 2 to 3)
and is only recommended for debugging.

The exporter and importer are completely symmetric in implementation: For
each encoding routine there is the matching and symmetric decoding routine.
This symmetry makes it very easy to change or extend the format: If a new
field needs to be encoded, a symmetric change can be made to exporter and
importer.
*/

package gc

import (
	"bytes"
	"cmd/compile/internal/big"
	"cmd/internal/obj"
	"encoding/binary"
	"fmt"
	"sort"
	"strings"
)

// debugging support
const (
	debugFormat = false // use debugging format for export data (emits a lot of additional data)
)

const exportVersion = "v0"

// Set forceNewExport to force the use of the new export format - for testing on the build dashboard.
// TODO(gri) remove eventually
const forceNewExport = false

// Export writes the export data for localpkg to out and returns the number of bytes written.
func Export(out *obj.Biobuf, trace bool) int {
	p := exporter{
		out:      out,
		pkgIndex: make(map[*Pkg]int),
		typIndex: make(map[*Type]int),
		trace:    trace,
	}

	// write low-level encoding format
	var format byte = 'c' // compact
	if debugFormat {
		format = 'd'
	}
	p.byte(format)

	// --- generic export data ---

	if p.trace {
		p.tracef("\n--- generic export data ---\n")
		if p.indent != 0 {
			Fatalf("incorrect indentation %d", p.indent)
		}
	}

	p.string(exportVersion)
	if p.trace {
		p.tracef("\n")
	}

	// populate type map with predeclared "known" types
	predecl := predeclared()
	for index, typ := range predecl {
		p.typIndex[typ] = index
	}
	if len(p.typIndex) != len(predecl) {
		Fatalf("duplicate entries in type map?")
	}

	// write package data
	if localpkg.Path != "" {
		Fatalf("local package path not empty: %q", localpkg.Path)
	}
	p.pkg(localpkg)

	// write compiler-specific flags
	// go.y:import_safety
	{
		var flags string
		if safemode != 0 {
			flags = "safe"
		}
		p.string(flags)
	}

	if p.trace {
		p.tracef("\n")
	}

	// collect objects to export
	var consts, vars, funcs []*Sym
	var types []*Type
	for _, n := range exportlist {
		sym := n.Sym
		// TODO(gri) Closures appear marked as exported.
		// Investigate and determine if we need this.
		if sym.Flags&SymExported != 0 {
			continue
		}
		sym.Flags |= SymExported

		// TODO(gri) Closures have dots in their names;
		// e.g., TestFloatZeroValue.func1 in math/big tests.
		// We may not need this eventually. See also comment
		// on sym.Flags&SymExported test above.
		if strings.Contains(sym.Name, ".") {
			Fatalf("unexpected export symbol: %v", sym)
		}

		if sym.Flags&SymExport != 0 {
			if sym.Def == nil {
				Fatalf("unknown export symbol: %v", sym)
			}
			switch n := sym.Def; n.Op {
			case OLITERAL:
				// constant
				typecheck(&n, Erv)
				if n == nil || n.Op != OLITERAL {
					Fatalf("dumpexportconst: oconst nil: %v", sym)
				}
				consts = append(consts, sym)

			case ONAME:
				// variable or function
				typecheck(&n, Erv|Ecall)
				if n == nil || n.Type == nil {
					Fatalf("variable/function exported but not defined: %v", sym)
				}
				if n.Type.Etype == TFUNC && n.Class == PFUNC {
					funcs = append(funcs, sym)
				} else {
					vars = append(vars, sym)
				}

			case OTYPE:
				// named type
				t := n.Type
				if t.Etype == TFORW {
					Fatalf("export of incomplete type %v", sym)
				}
				types = append(types, t)

			default:
				Fatalf("unexpected export symbol: %v %v", Oconv(int(n.Op), 0), sym)
			}
		}
	}
	exportlist = nil // match export.go use of exportlist

	// for reproducible output
	sort.Sort(symByName(consts))
	sort.Sort(symByName(vars))
	sort.Sort(symByName(funcs))
	// sort types later when we have fewer types left

	// write consts
	p.int(len(consts))
	for _, sym := range consts {
		n := sym.Def
		typ := n.Type // may or may not be specified
		// Untyped (ideal) constants get their own type. This decouples
		// the constant type from the encoding of the constant value.
		if typ == nil || isideal(typ) {
			typ = untype(n.Val().Ctype())
		}

		p.string(sym.Name)
		p.typ(typ)
		p.value(n.Val())
	}

	// write vars
	p.int(len(vars))
	for _, sym := range vars {
		p.string(sym.Name)
		p.typ(sym.Def.Type)
	}

	// write funcs
	p.int(len(funcs))
	for _, sym := range funcs {
		p.string(sym.Name)
		// The type can only be a signature for functions. However, by always
		// writing the complete type specification (rather than just a signature)
		// we keep the option open of sharing common signatures across multiple
		// functions as a means to further compress the export data.
		p.typ(sym.Def.Type)
		p.int(p.collectInlined(sym.Def))
	}

	// determine which types are still left to write and sort them
	i := 0
	for _, t := range types {
		if _, ok := p.typIndex[t]; !ok {
			types[i] = t
			i++
		}
	}
	types = types[:i]
	sort.Sort(typByName(types))

	// write types
	p.int(len(types))
	for _, t := range types {
		// Writing a type may further reduce the number of types
		// that are left to be written, but at this point we don't
		// care.
		p.typ(t)
	}

	if p.trace {
		p.tracef("\n")
	}

	// --- compiler-specific export data ---

	if p.trace {
		p.tracef("\n--- compiler specific export data ---\n")
		if p.indent != 0 {
			Fatalf("incorrect indentation")
		}
	}

	// write inlined function bodies
	p.int(len(p.inlined))
	for i, f := range p.inlined {
		p.body(i, f)
	}

	if p.trace {
		p.tracef("\n")
	}

	// --- end of export data ---

	return p.written
}

type symByName []*Sym

func (a symByName) Len() int           { return len(a) }
func (a symByName) Less(i, j int) bool { return a[i].Name < a[j].Name }
func (a symByName) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }

type typByName []*Type

func (a typByName) Len() int           { return len(a) }
func (a typByName) Less(i, j int) bool { return a[i].Sym.Name < a[j].Sym.Name }
func (a typByName) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }

type exporter struct {
	out      *obj.Biobuf
	pkgIndex map[*Pkg]int
	typIndex map[*Type]int
	inlined  []*Func

	written int // bytes written
	indent  int // for p.trace
	trace   bool
}

func (p *exporter) pkg(pkg *Pkg) {
	if pkg == nil {
		Fatalf("unexpected nil pkg")
	}

	// if we saw the package before, write its index (>= 0)
	if i, ok := p.pkgIndex[pkg]; ok {
		p.index('P', i)
		return
	}

	// otherwise, remember the package, write the package tag (< 0) and package data
	if p.trace {
		p.tracef("P%d = { ", len(p.pkgIndex))
		defer p.tracef("} ")
	}
	p.pkgIndex[pkg] = len(p.pkgIndex)

	p.tag(packageTag)
	p.string(pkg.Name)
	p.string(pkg.Path)
}

func (p *exporter) typ(t *Type) {
	if t == nil {
		Fatalf("nil type")
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
	if p.trace {
		p.tracef("T%d = {>\n", len(p.typIndex))
		defer p.tracef("<\n} ")
	}
	p.typIndex[t] = len(p.typIndex)

	// pick off named types
	if sym := t.Sym; sym != nil {
		// Fields should be exported by p.field().
		if t.Etype == TFIELD {
			Fatalf("printing a field/parameter with wrong function")
		}
		// Predeclared types should have been found in the type map.
		if t.Orig == t {
			Fatalf("predeclared type missing from type map?")
		}
		// TODO(gri) The assertion below seems incorrect (crashes during all.bash).
		// Investigate.
		/*
			// we expect the respective definition to point to us
			if sym.Def.Type != t {
				Fatalf("type definition doesn't point to us?")
			}
		*/

		p.tag(namedTag)
		p.qualifiedName(sym)

		// write underlying type
		p.typ(t.Orig)

		// interfaces don't have associated methods
		if t.Orig.Etype == TINTER {
			return
		}

		// sort methods for reproducible export format
		// TODO(gri) Determine if they are already sorted
		// in which case we can drop this step.
		var methods []*Type
		for m := t.Method; m != nil; m = m.Down {
			methods = append(methods, m)
		}
		sort.Sort(methodbyname(methods))
		p.int(len(methods))

		if p.trace && t.Method != nil {
			p.tracef("associated methods {>\n")
		}

		for _, m := range methods {
			p.string(m.Sym.Name)
			p.paramList(getthisx(m.Type))
			p.paramList(getinargx(m.Type))
			p.paramList(getoutargx(m.Type))
			p.int(p.collectInlined(m.Type.Nname))

			if p.trace && m.Down != nil {
				p.tracef("\n")
			}
		}

		if p.trace && t.Method != nil {
			p.tracef("<\n} ")
		}

		return
	}

	// otherwise we have a type literal
	switch t.Etype {
	case TARRAY:
		// TODO(gri) define named constant for the -100
		if t.Bound >= 0 || t.Bound == -100 {
			p.tag(arrayTag)
			p.int64(t.Bound)
		} else {
			p.tag(sliceTag)
		}
		p.typ(t.Type)

	case T_old_DARRAY:
		// see p.param use of T_old_DARRAY
		p.tag(dddTag)
		p.typ(t.Type)

	case TSTRUCT:
		p.tag(structTag)
		p.fieldList(t)

	case TPTR32, TPTR64: // could use Tptr but these are constants
		p.tag(pointerTag)
		p.typ(t.Type)

	case TFUNC:
		p.tag(signatureTag)
		p.paramList(getinargx(t))
		p.paramList(getoutargx(t))

	case TINTER:
		p.tag(interfaceTag)

		// gc doesn't separate between embedded interfaces
		// and methods declared explicitly with an interface
		p.int(0) // no embedded interfaces
		p.methodList(t)

	case TMAP:
		p.tag(mapTag)
		p.typ(t.Down) // key
		p.typ(t.Type) // val

	case TCHAN:
		p.tag(chanTag)
		p.int(int(t.Chan))
		p.typ(t.Type)

	default:
		Fatalf("unexpected type: %s (Etype = %d)", Tconv(t, 0), t.Etype)
	}
}

func (p *exporter) qualifiedName(sym *Sym) {
	p.string(sym.Name)
	p.pkg(sym.Pkg)
}

func (p *exporter) fieldList(t *Type) {
	if p.trace && t.Type != nil {
		p.tracef("fields {>\n")
		defer p.tracef("<\n} ")
	}

	p.int(countfield(t))
	for f := t.Type; f != nil; f = f.Down {
		p.field(f)
		if p.trace && f.Down != nil {
			p.tracef("\n")
		}
	}
}

func (p *exporter) field(f *Type) {
	if f.Etype != TFIELD {
		Fatalf("field expected")
	}

	p.fieldName(f)
	p.typ(f.Type)
	p.note(f.Note)
}

func (p *exporter) note(n *string) {
	var s string
	if n != nil {
		s = *n
	}
	p.string(s)
}

func (p *exporter) methodList(t *Type) {
	if p.trace && t.Type != nil {
		p.tracef("methods {>\n")
		defer p.tracef("<\n} ")
	}

	p.int(countfield(t))
	for m := t.Type; m != nil; m = m.Down {
		p.method(m)
		if p.trace && m.Down != nil {
			p.tracef("\n")
		}
	}
}

func (p *exporter) method(m *Type) {
	if m.Etype != TFIELD {
		Fatalf("method expected")
	}

	p.fieldName(m)
	// TODO(gri) For functions signatures, we use p.typ() to export
	// so we could share the same type with multiple functions. Do
	// the same here, or never try to do this for functions.
	p.paramList(getinargx(m.Type))
	p.paramList(getoutargx(m.Type))
}

// fieldName is like qualifiedName but it doesn't record the package
// for blank (_) or exported names.
func (p *exporter) fieldName(t *Type) {
	sym := t.Sym

	var name string
	if t.Embedded == 0 {
		name = sym.Name
	} else if bname := basetypeName(t); bname != "" && !exportname(bname) {
		// anonymous field with unexported base type name: use "?" as field name
		// (bname != "" per spec, but we are conservative in case of errors)
		name = "?"
	}

	p.string(name)
	if name == "?" || name != "_" && name != "" && !exportname(name) {
		p.pkg(sym.Pkg)
	}
}

func basetypeName(t *Type) string {
	s := t.Sym
	if s == nil && Isptr[t.Etype] {
		s = t.Type.Sym // deref
	}
	if s != nil {
		return s.Name
	}
	return ""
}

func (p *exporter) paramList(params *Type) {
	if params.Etype != TSTRUCT || !params.Funarg {
		Fatalf("parameter list expected")
	}

	// use negative length to indicate unnamed parameters
	// (look at the first parameter only since either all
	// names are present or all are absent)
	n := countfield(params)
	if n > 0 && parName(params.Type) == "" {
		n = -n
	}
	p.int(n)
	for q := params.Type; q != nil; q = q.Down {
		p.param(q, n)
	}
}

func (p *exporter) param(q *Type, n int) {
	if q.Etype != TFIELD {
		Fatalf("parameter expected")
	}
	t := q.Type
	if q.Isddd {
		// create a fake type to encode ... just for the p.typ call
		// (T_old_DARRAY is not used anywhere else in the compiler,
		// we use it here to communicate between p.param and p.typ.)
		t = &Type{Etype: T_old_DARRAY, Type: t.Type}
	}
	p.typ(t)
	if n > 0 {
		p.string(parName(q))
	}
	// TODO(gri) This is compiler-specific (escape info).
	// Move into compiler-specific section eventually?
	// (Not having escape info causes tests to fail, e.g. runtime GCInfoTest)
	p.note(q.Note)
}

func parName(q *Type) string {
	if q.Sym == nil {
		return ""
	}
	name := q.Sym.Name
	// undo gc-internal name mangling - we just need the source name
	if len(name) > 0 && name[0] == '~' {
		// name is ~b%d or ~r%d
		switch name[1] {
		case 'b':
			return "_"
		case 'r':
			return ""
		default:
			Fatalf("unexpected parameter name: %s", name)
		}
	}
	// undo gc-internal name specialization
	if i := strings.Index(name, "·"); i > 0 {
		name = name[:i] // cut off numbering
	}
	return name
}

func (p *exporter) value(x Val) {
	if p.trace {
		p.tracef("= ")
	}

	switch x := x.U.(type) {
	case bool:
		tag := falseTag
		if x {
			tag = trueTag
		}
		p.tag(tag)

	case *Mpint:
		if Mpcmpfixfix(Minintval[TINT64], x) <= 0 && Mpcmpfixfix(x, Maxintval[TINT64]) <= 0 {
			// common case: x fits into an int64 - use compact encoding
			p.tag(int64Tag)
			p.int64(Mpgetfix(x))
			return
		}
		// uncommon case: large x - use float encoding
		// (powers of 2 will be encoded efficiently with exponent)
		p.tag(floatTag)
		f := newMpflt()
		Mpmovefixflt(f, x)
		p.float(f)

	case *Mpflt:
		p.tag(floatTag)
		p.float(x)

	case *Mpcplx:
		p.tag(complexTag)
		p.float(&x.Real)
		p.float(&x.Imag)

	case string:
		p.tag(stringTag)
		p.string(x)

	default:
		Fatalf("unexpected value %v (%T)", x, x)
	}
}

func (p *exporter) float(x *Mpflt) {
	// extract sign (there is no -0)
	f := &x.Val
	sign := f.Sign()
	if sign == 0 {
		// x == 0
		p.int(0)
		return
	}
	// x != 0

	// extract exponent such that 0.5 <= m < 1.0
	var m big.Float
	exp := f.MantExp(&m)

	// extract mantissa as *big.Int
	// - set exponent large enough so mant satisfies mant.IsInt()
	// - get *big.Int from mant
	m.SetMantExp(&m, int(m.MinPrec()))
	mant, acc := m.Int(nil)
	if acc != big.Exact {
		Fatalf("internal error")
	}

	p.int(sign)
	p.int(exp)
	p.string(string(mant.Bytes()))
}

// ----------------------------------------------------------------------------
// Inlined function bodies

// TODO(gri) This section is incomplete. At the moment nothing meaningful
// is written out for exported functions with inlined function bodies.

func (p *exporter) collectInlined(n *Node) int {
	if n != nil && n.Func != nil && n.Func.Inl != nil {
		// when lazily typechecking inlined bodies, some re-exported ones may not have been typechecked yet.
		// currently that can leave unresolved ONONAMEs in import-dot-ed packages in the wrong package
		if Debug['l'] < 2 {
			typecheckinl(n)
		}
		p.inlined = append(p.inlined, n.Func)
		return len(p.inlined) - 1 // index >= 0 => inlined
	}
	return -1 // index < 0 => not inlined
}

func (p *exporter) body(i int, f *Func) {
	p.int(i)
	p.block(f.Inl)
}

func (p *exporter) block(list *NodeList) {
	p.int(count(list))
	for q := list; q != nil; q = q.Next {
		p.stmt(q.N)
	}
}

func (p *exporter) stmt(n *Node) {
	// TODO(gri) do something sensible here
	p.string("body")
}

// ----------------------------------------------------------------------------
// Low-level encoders

func (p *exporter) index(marker byte, index int) {
	if index < 0 {
		Fatalf("invalid index < 0")
	}
	if debugFormat {
		p.marker('t')
	}
	if p.trace {
		p.tracef("%c%d ", marker, index)
	}
	p.rawInt64(int64(index))
}

func (p *exporter) tag(tag int) {
	if tag >= 0 {
		Fatalf("invalid tag >= 0")
	}
	if debugFormat {
		p.marker('t')
	}
	if p.trace {
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
	if p.trace {
		p.tracef("%d ", x)
	}
	p.rawInt64(x)
}

func (p *exporter) string(s string) {
	if debugFormat {
		p.marker('s')
	}
	if p.trace {
		p.tracef("%q ", s)
	}
	p.rawInt64(int64(len(s)))
	for i := 0; i < len(s); i++ {
		p.byte(s[i])
	}
}

// marker emits a marker byte and position information which makes
// it easy for a reader to detect if it is "out of sync". Used for
// debugFormat format only.
func (p *exporter) marker(m byte) {
	p.byte(m)
	p.rawInt64(int64(p.written))
}

// rawInt64 should only be used by low-level encoders
func (p *exporter) rawInt64(x int64) {
	var tmp [binary.MaxVarintLen64]byte
	n := binary.PutVarint(tmp[:], x)
	for i := 0; i < n; i++ {
		p.byte(tmp[i])
	}
}

// byte is the bottleneck interface to write to p.out.
// byte escapes b as follows (any encoding does that
// hides '$'):
//
//	'$'  => '|' 'S'
//	'|'  => '|' '|'
//
// Necessary so other tools can find the end of the
// export data by searching for "$$".
func (p *exporter) byte(b byte) {
	switch b {
	case '$':
		// write '$' as '|' 'S'
		b = 'S'
		fallthrough
	case '|':
		// write '|' as '|' '|'
		obj.Bputc(p.out, '|')
		p.written++
	}
	obj.Bputc(p.out, b)
	p.written++
}

// tracef is like fmt.Printf but it rewrites the format string
// to take care of indentation.
func (p *exporter) tracef(format string, args ...interface{}) {
	if strings.IndexAny(format, "<>\n") >= 0 {
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

// ----------------------------------------------------------------------------
// Export format

// Tags. Must be < 0.
const (
	// Packages
	packageTag = -(iota + 1)

	// Types
	namedTag
	arrayTag
	sliceTag
	dddTag
	structTag
	pointerTag
	signatureTag
	interfaceTag
	mapTag
	chanTag

	// Values
	falseTag
	trueTag
	int64Tag
	floatTag
	fractionTag // not used by gc
	complexTag
	stringTag
)

// Debugging support.
// (tagString is only used when tracing is enabled)
var tagString = [...]string{
	// Packages:
	-packageTag: "package",

	// Types:
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

	// Values:
	-falseTag:    "false",
	-trueTag:     "true",
	-int64Tag:    "int64",
	-floatTag:    "float",
	-fractionTag: "fraction",
	-complexTag:  "complex",
	-stringTag:   "string",
}

// untype returns the "pseudo" untyped type for a Ctype (import/export use only).
// (we can't use an pre-initialized array because we must be sure all types are
// set up)
func untype(ctype Ctype) *Type {
	switch ctype {
	case CTINT:
		return idealint
	case CTRUNE:
		return idealrune
	case CTFLT:
		return idealfloat
	case CTCPLX:
		return idealcomplex
	case CTSTR:
		return idealstring
	case CTBOOL:
		return idealbool
	case CTNIL:
		return Types[TNIL]
	}
	Fatalf("unknown Ctype")
	return nil
}

var (
	idealint     = typ(TIDEAL)
	idealrune    = typ(TIDEAL)
	idealfloat   = typ(TIDEAL)
	idealcomplex = typ(TIDEAL)
)

var predecl []*Type // initialized lazily

func predeclared() []*Type {
	if predecl == nil {
		// initialize lazily to be sure that all
		// elements have been initialized before
		predecl = []*Type{
			// basic types
			Types[TBOOL],
			Types[TINT],
			Types[TINT8],
			Types[TINT16],
			Types[TINT32],
			Types[TINT64],
			Types[TUINT],
			Types[TUINT8],
			Types[TUINT16],
			Types[TUINT32],
			Types[TUINT64],
			Types[TUINTPTR],
			Types[TFLOAT32],
			Types[TFLOAT64],
			Types[TCOMPLEX64],
			Types[TCOMPLEX128],
			Types[TSTRING],

			// aliases
			bytetype,
			runetype,

			// error
			errortype,

			// untyped types
			untype(CTBOOL),
			untype(CTINT),
			untype(CTRUNE),
			untype(CTFLT),
			untype(CTCPLX),
			untype(CTSTR),
			untype(CTNIL),

			// package unsafe
			Types[TUNSAFEPTR],
		}
	}
	return predecl
}

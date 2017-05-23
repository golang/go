// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Binary package export.
// (see fmt.go, parser.go as "documentation" for how to use/setup data structures)

/*
1) Export data encoding principles:

The export data is a serialized description of the graph of exported
"objects": constants, types, variables, and functions. In general,
types - but also objects referred to from inlined function bodies -
can be reexported and so we need to know which package they are coming
from. Therefore, packages are also part of the export graph.

The roots of the graph are two lists of objects. The 1st list (phase 1,
see Export) contains all objects that are exported at the package level.
These objects are the full representation of the package's API, and they
are the only information a platform-independent tool (e.g., go/types)
needs to know to type-check against a package.

The 2nd list of objects contains all objects referred to from exported
inlined function bodies. These objects are needed by the compiler to
make sense of the function bodies; the exact list contents are compiler-
specific.

Finally, the export data contains a list of representations for inlined
function bodies. The format of this representation is compiler specific.

The graph is serialized in in-order fashion, starting with the roots.
Each object in the graph is serialized by writing its fields sequentially.
If the field is a pointer to another object, that object is serialized,
recursively. Otherwise the field is written. Non-pointer fields are all
encoded as integer or string values.

Some objects (packages, types) may be referred to more than once. When
reaching an object that was not serialized before, an integer _index_
is assigned to it, starting at 0. In this case, the encoding starts
with an integer _tag_ < 0. The tag value indicates the kind of object
that follows and that this is the first time that we see this object.
If the object was already serialized, the encoding is simply the object
index >= 0. An importer can trivially determine if an object needs to
be read in for the first time (tag < 0) and entered into the respective
object table, or if the object was seen already (index >= 0), in which
case the index is used to look up the object in a table.

Before exporting or importing, the type tables are populated with the
predeclared types (int, string, error, unsafe.Pointer, etc.). This way
they are automatically encoded with a known and fixed type index.

2) Encoding format:

The export data starts with a single byte indicating the encoding format
(compact, or with debugging information), followed by a version string
(so we can evolve the encoding if need be), and then the package object
for the exported package (with an empty path).

After this header, two lists of objects and the list of inlined function
bodies follows.

The encoding of objects is straight-forward: Constants, variables, and
functions start with their name, type, and possibly a value. Named types
record their name and package so that they can be canonicalized: If the
same type was imported before via another import, the importer must use
the previously imported type pointer so that we have exactly one version
(i.e., one pointer) for each named type (and read but discard the current
type encoding). Unnamed types simply encode their respective fields.

In the encoding, some lists start with the list length. Some lists are
terminated with an end marker (usually for lists where we may not know
the length a priori).

Integers use variable-length encoding for compact representation.

Strings are canonicalized similar to objects that may occur multiple times:
If the string was exported already, it is represented by its index only.
Otherwise, the export data starts with the negative string length (negative,
so we can distinguish from string index), followed by the string bytes.
The empty string is mapped to index 0.

The exporter and importer are completely symmetric in implementation: For
each encoding routine there is a matching and symmetric decoding routine.
This symmetry makes it very easy to change or extend the format: If a new
field needs to be encoded, a symmetric change can be made to exporter and
importer.

3) Making changes to the encoding format:

Any change to the encoding format requires a respective change in the
exporter below and a corresponding symmetric change to the importer in
bimport.go.

Furthermore, it requires a corresponding change to go/internal/gcimporter
and golang.org/x/tools/go/gcimporter15. Changes to the latter must preserve
compatibility with both the last release of the compiler, and with the
corresponding compiler at tip. That change is necessarily more involved,
as it must switch based on the version number in the export data file.

It is recommended to turn on debugFormat when working on format changes
as it will help finding encoding/decoding inconsistencies quickly.

Special care must be taken to update builtin.go when the export format
changes: builtin.go contains the export data obtained by compiling the
builtin/runtime.go and builtin/unsafe.go files; those compilations in
turn depend on importing the data in builtin.go. Thus, when the export
data format changes, the compiler must be able to import the data in
builtin.go even if its format has not yet changed. Proceed in several
steps as follows:

- Change the exporter to use the new format, and use a different version
  string as well.
- Update the importer accordingly, but accept both the old and the new
  format depending on the version string.
- all.bash should pass at this point.
- Run mkbuiltin.go: this will create a new builtin.go using the new
  export format.
- go test -run Builtin should pass at this point.
- Remove importer support for the old export format and (maybe) revert
  the version string again (it's only needed to mark the transition).
- all.bash should still pass.

Don't forget to set debugFormat to false.
*/

package gc

import (
	"bufio"
	"bytes"
	"cmd/compile/internal/big"
	"encoding/binary"
	"fmt"
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

// forceObjFileStability enforces additional constraints in export data
// and other parts of the compiler to eliminate object file differences
// only due to the choice of export format.
// TODO(gri) disable and remove once there is only one export format again
const forceObjFileStability = true

// Supported export format versions.
// TODO(gri) Make this more systematic (issue #16244).
const (
	exportVersion0 = "v0"
	exportVersion1 = "v1"
	exportVersion  = exportVersion1
)

// exportInlined enables the export of inlined function bodies and related
// dependencies. The compiler should work w/o any loss of functionality with
// the flag disabled, but the generated code will lose access to inlined
// function bodies across packages, leading to performance bugs.
// Leave for debugging.
const exportInlined = true // default: true

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
	out *bufio.Writer

	// object -> index maps, indexed in order of serialization
	strIndex map[string]int
	pkgIndex map[*Pkg]int
	typIndex map[*Type]int
	funcList []*Func

	// position encoding
	posInfoFormat bool
	prevFile      string
	prevLine      int

	// debugging support
	written int // bytes written
	indent  int // for p.trace
	trace   bool
}

// export writes the exportlist for localpkg to out and returns the number of bytes written.
func export(out *bufio.Writer, trace bool) int {
	p := exporter{
		out:      out,
		strIndex: map[string]int{"": 0}, // empty string is mapped to 0
		pkgIndex: make(map[*Pkg]int),
		typIndex: make(map[*Type]int),
		// don't emit pos info for builtin packages
		// (not needed and avoids path name diffs in builtin.go between
		// Windows and non-Windows machines, exposed via builtin_test.go)
		posInfoFormat: Debug['A'] == 0,
		trace:         trace,
	}

	// TODO(gri) clean up the ad-hoc encoding of the file format below
	// (we need this so we can read the builtin package export data
	// easily w/o being affected by format changes)

	// first byte indicates low-level encoding format
	var format byte = 'c' // compact
	if debugFormat {
		format = 'd'
	}
	p.rawByte(format)

	format = 'n' // track named types only
	if trackAllTypes {
		format = 'a'
	}
	p.rawByte(format)

	// posInfo exported or not?
	p.bool(p.posInfoFormat)

	// --- generic export data ---

	if p.trace {
		p.tracef("\n--- package ---\n")
		if p.indent != 0 {
			Fatalf("exporter: incorrect indentation %d", p.indent)
		}
	}

	if p.trace {
		p.tracef("version = ")
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
		Fatalf("exporter: duplicate entries in type map?")
	}

	// write package data
	if localpkg.Path != "" {
		Fatalf("exporter: local package path not empty: %q", localpkg.Path)
	}
	p.pkg(localpkg)
	if p.trace {
		p.tracef("\n")
	}

	// export objects
	//
	// First, export all exported (package-level) objects; i.e., all objects
	// in the current exportlist. These objects represent all information
	// required to import this package and type-check against it; i.e., this
	// is the platform-independent export data. The format is generic in the
	// sense that different compilers can use the same representation.
	//
	// During this first phase, more objects may be added to the exportlist
	// (due to inlined function bodies and their dependencies). Export those
	// objects in a second phase. That data is platform-specific as it depends
	// on the inlining decisions of the compiler and the representation of the
	// inlined function bodies.

	// remember initial exportlist length
	var numglobals = len(exportlist)

	// Phase 1: Export objects in _current_ exportlist; exported objects at
	//          package level.
	// Use range since we want to ignore objects added to exportlist during
	// this phase.
	objcount := 0
	for _, n := range exportlist {
		sym := n.Sym

		if sym.Flags&SymExported != 0 {
			continue
		}
		sym.Flags |= SymExported

		// TODO(gri) Closures have dots in their names;
		// e.g., TestFloatZeroValue.func1 in math/big tests.
		if strings.Contains(sym.Name, ".") {
			Fatalf("exporter: unexpected symbol: %v", sym)
		}

		// TODO(gri) Should we do this check?
		// if sym.Flags&SymExport == 0 {
		// 	continue
		// }

		if sym.Def == nil {
			Fatalf("exporter: unknown export symbol: %v", sym)
		}

		// TODO(gri) Optimization: Probably worthwhile collecting
		// long runs of constants and export them "in bulk" (saving
		// tags and types, and making import faster).

		if p.trace {
			p.tracef("\n")
		}
		p.obj(sym)
		objcount++
	}

	// indicate end of list
	if p.trace {
		p.tracef("\n")
	}
	p.tag(endTag)

	// for self-verification only (redundant)
	p.int(objcount)

	// --- compiler-specific export data ---

	if p.trace {
		p.tracef("\n--- compiler-specific export data ---\n[ ")
		if p.indent != 0 {
			Fatalf("exporter: incorrect indentation")
		}
	}

	// write compiler-specific flags
	p.bool(safemode)
	if p.trace {
		p.tracef("\n")
	}

	// Phase 2: Export objects added to exportlist during phase 1.
	// Don't use range since exportlist may grow during this phase
	// and we want to export all remaining objects.
	objcount = 0
	for i := numglobals; exportInlined && i < len(exportlist); i++ {
		n := exportlist[i]
		sym := n.Sym

		// TODO(gri) The rest of this loop body is identical with
		// the loop body above. Leave alone for now since there
		// are different optimization opportunities, but factor
		// eventually.

		if sym.Flags&SymExported != 0 {
			continue
		}
		sym.Flags |= SymExported

		// TODO(gri) Closures have dots in their names;
		// e.g., TestFloatZeroValue.func1 in math/big tests.
		if strings.Contains(sym.Name, ".") {
			Fatalf("exporter: unexpected symbol: %v", sym)
		}

		// TODO(gri) Should we do this check?
		// if sym.Flags&SymExport == 0 {
		// 	continue
		// }

		if sym.Def == nil {
			Fatalf("exporter: unknown export symbol: %v", sym)
		}

		// TODO(gri) Optimization: Probably worthwhile collecting
		// long runs of constants and export them "in bulk" (saving
		// tags and types, and making import faster).

		if p.trace {
			p.tracef("\n")
		}
		p.obj(sym)
		objcount++
	}

	// indicate end of list
	if p.trace {
		p.tracef("\n")
	}
	p.tag(endTag)

	// for self-verification only (redundant)
	p.int(objcount)

	// --- inlined function bodies ---

	if p.trace {
		p.tracef("\n--- inlined function bodies ---\n")
		if p.indent != 0 {
			Fatalf("exporter: incorrect indentation")
		}
	}

	// write inlineable function bodies
	objcount = 0
	for i, f := range p.funcList {
		if f != nil {
			// function has inlineable body:
			// write index and body
			if p.trace {
				p.tracef("\n----\nfunc { %s }\n", hconv(f.Inl, FmtSharp))
			}
			p.int(i)
			p.stmtList(f.Inl)
			if p.trace {
				p.tracef("\n")
			}
			objcount++
		}
	}

	// indicate end of list
	if p.trace {
		p.tracef("\n")
	}
	p.int(-1) // invalid index terminates list

	// for self-verification only (redundant)
	p.int(objcount)

	if p.trace {
		p.tracef("\n--- end ---\n")
	}

	// --- end of export data ---

	return p.written
}

func (p *exporter) pkg(pkg *Pkg) {
	if pkg == nil {
		Fatalf("exporter: unexpected nil pkg")
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

func unidealType(typ *Type, val Val) *Type {
	// Untyped (ideal) constants get their own type. This decouples
	// the constant type from the encoding of the constant value.
	if typ == nil || typ.IsUntyped() {
		typ = untype(val.Ctype())
	}
	return typ
}

func (p *exporter) obj(sym *Sym) {
	// Exported objects may be from different packages because they
	// may be re-exported as depencies when exporting inlined function
	// bodies. Thus, exported object names must be fully qualified.
	//
	// TODO(gri) This can only happen if exportInlined is enabled
	// (default), and during phase 2 of object export. Objects exported
	// in phase 1 (compiler-indendepent objects) are by definition only
	// the objects from the current package and not pulled in via inlined
	// function bodies. In that case the package qualifier is not needed.
	// Possible space optimization.

	n := sym.Def
	switch n.Op {
	case OLITERAL:
		// constant
		// TODO(gri) determine if we need the typecheck call here
		n = typecheck(n, Erv)
		if n == nil || n.Op != OLITERAL {
			Fatalf("exporter: dumpexportconst: oconst nil: %v", sym)
		}

		p.tag(constTag)
		p.pos(n)
		// TODO(gri) In inlined functions, constants are used directly
		// so they should never occur as re-exported objects. We may
		// not need the qualified name here. See also comment above.
		// Possible space optimization.
		p.qualifiedName(sym)
		p.typ(unidealType(n.Type, n.Val()))
		p.value(n.Val())

	case OTYPE:
		// named type
		t := n.Type
		if t.Etype == TFORW {
			Fatalf("exporter: export of incomplete type %v", sym)
		}

		p.tag(typeTag)
		p.typ(t)

	case ONAME:
		// variable or function
		n = typecheck(n, Erv|Ecall)
		if n == nil || n.Type == nil {
			Fatalf("exporter: variable/function exported but not defined: %v", sym)
		}

		if n.Type.Etype == TFUNC && n.Class == PFUNC {
			// function
			p.tag(funcTag)
			p.pos(n)
			p.qualifiedName(sym)

			sig := sym.Def.Type
			inlineable := isInlineable(sym.Def)

			p.paramList(sig.Params(), inlineable)
			p.paramList(sig.Results(), inlineable)

			var f *Func
			if inlineable {
				f = sym.Def.Func
				// TODO(gri) re-examine reexportdeplist:
				// Because we can trivially export types
				// in-place, we don't need to collect types
				// inside function bodies in the exportlist.
				// With an adjusted reexportdeplist used only
				// by the binary exporter, we can also avoid
				// the global exportlist.
				reexportdeplist(f.Inl)
			}
			p.funcList = append(p.funcList, f)
		} else {
			// variable
			p.tag(varTag)
			p.pos(n)
			p.qualifiedName(sym)
			p.typ(sym.Def.Type)
		}

	default:
		Fatalf("exporter: unexpected export symbol: %v %v", n.Op, sym)
	}
}

func (p *exporter) pos(n *Node) {
	if !p.posInfoFormat {
		return
	}

	file, line := fileLine(n)
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

func fileLine(n *Node) (file string, line int) {
	if n != nil {
		file, line = Ctxt.LineHist.AbsFileLine(int(n.Lineno))
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

func isInlineable(n *Node) bool {
	if exportInlined && n != nil && n.Func != nil && n.Func.Inl.Len() != 0 {
		// when lazily typechecking inlined bodies, some re-exported ones may not have been typechecked yet.
		// currently that can leave unresolved ONONAMEs in import-dot-ed packages in the wrong package
		if Debug['l'] < 2 {
			typecheckinl(n)
		}
		return true
	}
	return false
}

var errorInterface *Type // lazily initialized

func (p *exporter) typ(t *Type) {
	if t == nil {
		Fatalf("exporter: nil type")
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
		if p.trace {
			p.tracef("T%d = {>\n", len(p.typIndex))
			defer p.tracef("<\n} ")
		}
		p.typIndex[t] = len(p.typIndex)
	}

	// pick off named types
	if tsym := t.Sym; tsym != nil {
		if !trackAllTypes {
			// if we don't track all types, track named types now
			p.typIndex[t] = len(p.typIndex)
		}

		// Predeclared types should have been found in the type map.
		if t.Orig == t {
			Fatalf("exporter: predeclared type missing from type map?")
		}

		n := typenod(t)
		if n.Type != t {
			Fatalf("exporter: named type definition incorrectly set up")
		}

		p.tag(namedTag)
		p.pos(n)
		p.qualifiedName(tsym)

		// write underlying type
		orig := t.Orig
		if orig == errortype {
			// The error type is the only predeclared type which has
			// a composite underlying type. When we encode that type,
			// make sure to encode the underlying interface rather than
			// the named type again. See also the comment in universe.go
			// regarding the errortype and issue #15920.
			if errorInterface == nil {
				errorInterface = makeErrorInterface()
			}
			orig = errorInterface
		}
		p.typ(orig)

		// interfaces don't have associated methods
		if t.Orig.IsInterface() {
			return
		}

		// sort methods for reproducible export format
		// TODO(gri) Determine if they are already sorted
		// in which case we can drop this step.
		var methods []*Field
		for _, m := range t.Methods().Slice() {
			methods = append(methods, m)
		}
		sort.Sort(methodbyname(methods))
		p.int(len(methods))

		if p.trace && len(methods) > 0 {
			p.tracef("associated methods {>")
		}

		for _, m := range methods {
			if p.trace {
				p.tracef("\n")
			}
			if strings.Contains(m.Sym.Name, ".") {
				Fatalf("invalid symbol name: %s (%v)", m.Sym.Name, m.Sym)
			}

			p.pos(m.Nname)
			p.fieldSym(m.Sym, false)

			sig := m.Type
			mfn := sig.Nname()
			inlineable := isInlineable(mfn)

			p.paramList(sig.Recvs(), inlineable)
			p.paramList(sig.Params(), inlineable)
			p.paramList(sig.Results(), inlineable)

			// for issue #16243
			// We make this conditional for 1.7 to avoid consistency problems
			// with installed packages compiled with an older version.
			// TODO(gri) Clean up after 1.7 is out (issue #16244)
			if exportVersion == exportVersion1 {
				p.bool(m.Nointerface)
			}

			var f *Func
			if inlineable {
				f = mfn.Func
				reexportdeplist(mfn.Func.Inl)
			}
			p.funcList = append(p.funcList, f)
		}

		if p.trace && len(methods) > 0 {
			p.tracef("<\n} ")
		}

		return
	}

	// otherwise we have a type literal
	switch t.Etype {
	case TARRAY:
		if t.isDDDArray() {
			Fatalf("array bounds should be known at export time: %v", t)
		}
		p.tag(arrayTag)
		p.int64(t.NumElem())
		p.typ(t.Elem())

	case TSLICE:
		p.tag(sliceTag)
		p.typ(t.Elem())

	case TDDDFIELD:
		// see p.param use of TDDDFIELD
		p.tag(dddTag)
		p.typ(t.DDDField())

	case TSTRUCT:
		p.tag(structTag)
		p.fieldList(t)

	case TPTR32, TPTR64: // could use Tptr but these are constants
		p.tag(pointerTag)
		p.typ(t.Elem())

	case TFUNC:
		p.tag(signatureTag)
		p.paramList(t.Params(), false)
		p.paramList(t.Results(), false)

	case TINTER:
		p.tag(interfaceTag)

		// gc doesn't separate between embedded interfaces
		// and methods declared explicitly with an interface
		p.int(0) // no embedded interfaces
		p.methodList(t)

	case TMAP:
		p.tag(mapTag)
		p.typ(t.Key())
		p.typ(t.Val())

	case TCHAN:
		p.tag(chanTag)
		p.int(int(t.ChanDir()))
		p.typ(t.Elem())

	default:
		Fatalf("exporter: unexpected type: %s (Etype = %d)", Tconv(t, 0), t.Etype)
	}
}

func (p *exporter) qualifiedName(sym *Sym) {
	if strings.Contains(sym.Name, ".") {
		Fatalf("exporter: invalid symbol name: %s", sym.Name)
	}
	p.string(sym.Name)
	p.pkg(sym.Pkg)
}

func (p *exporter) fieldList(t *Type) {
	if p.trace && t.NumFields() > 0 {
		p.tracef("fields {>")
		defer p.tracef("<\n} ")
	}

	p.int(t.NumFields())
	for _, f := range t.Fields().Slice() {
		if p.trace {
			p.tracef("\n")
		}
		p.field(f)
	}
}

func (p *exporter) field(f *Field) {
	p.pos(f.Nname)
	p.fieldName(f.Sym, f)
	p.typ(f.Type)
	p.string(f.Note)
}

func (p *exporter) methodList(t *Type) {
	if p.trace && t.NumFields() > 0 {
		p.tracef("methods {>")
		defer p.tracef("<\n} ")
	}

	p.int(t.NumFields())
	for _, m := range t.Fields().Slice() {
		if p.trace {
			p.tracef("\n")
		}
		p.method(m)
	}
}

func (p *exporter) method(m *Field) {
	p.pos(m.Nname)
	p.fieldName(m.Sym, m)
	p.paramList(m.Type.Params(), false)
	p.paramList(m.Type.Results(), false)
}

// fieldName is like qualifiedName but it doesn't record the package
// for blank (_) or exported names.
func (p *exporter) fieldName(sym *Sym, t *Field) {
	if t != nil && sym != t.Sym {
		Fatalf("exporter: invalid fieldName parameters")
	}

	name := sym.Name
	if t != nil {
		if t.Embedded == 0 {
			name = sym.Name
		} else if bname := basetypeName(t.Type); bname != "" && !exportname(bname) {
			// anonymous field with unexported base type name: use "?" as field name
			// (bname != "" per spec, but we are conservative in case of errors)
			name = "?"
		} else {
			name = ""
		}
	}

	if strings.Contains(name, ".") {
		Fatalf("exporter: invalid symbol name: %s", name)
	}
	p.string(name)
	if name == "?" || name != "_" && name != "" && !exportname(name) {
		p.pkg(sym.Pkg)
	}
}

func basetypeName(t *Type) string {
	s := t.Sym
	if s == nil && t.IsPtr() {
		s = t.Elem().Sym // deref
	}
	if s != nil {
		if strings.Contains(s.Name, ".") {
			Fatalf("exporter: invalid symbol name: %s", s.Name)
		}
		return s.Name
	}
	return ""
}

func (p *exporter) paramList(params *Type, numbered bool) {
	if !params.IsFuncArgStruct() {
		Fatalf("exporter: parameter list expected")
	}

	// use negative length to indicate unnamed parameters
	// (look at the first parameter only since either all
	// names are present or all are absent)
	//
	// TODO(gri) If we don't have an exported function
	// body, the parameter names are irrelevant for the
	// compiler (though they may be of use for other tools).
	// Possible space optimization.
	n := params.NumFields()
	if n > 0 && parName(params.Field(0), numbered) == "" {
		n = -n
	}
	p.int(n)
	for _, q := range params.Fields().Slice() {
		p.param(q, n, numbered)
	}
}

func (p *exporter) param(q *Field, n int, numbered bool) {
	t := q.Type
	if q.Isddd {
		// create a fake type to encode ... just for the p.typ call
		t = typDDDField(t.Elem())
	}
	p.typ(t)
	if n > 0 {
		name := parName(q, numbered)
		if name == "" {
			// Sometimes we see an empty name even for n > 0.
			// This appears to happen for interface methods
			// with _ (blank) parameter names. Make sure we
			// have a proper name and package so we don't crash
			// during import (see also issue #15470).
			// (parName uses "" instead of "?" as in fmt.go)
			// TODO(gri) review parameter name encoding
			name = "_"
		}
		p.string(name)
		if name != "_" {
			// Because of (re-)exported inlined functions
			// the importpkg may not be the package to which this
			// function (and thus its parameter) belongs. We need to
			// supply the parameter package here. We need the package
			// when the function is inlined so we can properly resolve
			// the name. The _ (blank) parameter cannot be accessed, so
			// we don't need to export a package.
			//
			// TODO(gri) This is compiler-specific. Try using importpkg
			// here and then update the symbols if we find an inlined
			// body only. Otherwise, the parameter name is ignored and
			// the package doesn't matter. This would remove an int
			// (likely 1 byte) for each named parameter.
			p.pkg(q.Sym.Pkg)
		}
	}
	// TODO(gri) This is compiler-specific (escape info).
	// Move into compiler-specific section eventually?
	// (Not having escape info causes tests to fail, e.g. runtime GCInfoTest)
	p.string(q.Note)
}

func parName(f *Field, numbered bool) string {
	s := f.Sym
	if s == nil {
		return ""
	}

	// Take the name from the original, lest we substituted it with ~r%d or ~b%d.
	// ~r%d is a (formerly) unnamed result.
	if f.Nname != nil {
		if f.Nname.Orig != nil {
			s = f.Nname.Orig.Sym
			if s != nil && s.Name[0] == '~' {
				if s.Name[1] == 'r' { // originally an unnamed result
					return "" // s = nil
				} else if s.Name[1] == 'b' { // originally the blank identifier _
					return "_" // belongs to localpkg
				}
			}
		} else {
			return "" // s = nil
		}
	}

	if s == nil {
		return ""
	}

	// print symbol with Vargen number or not as desired
	name := s.Name
	if strings.Contains(name, ".") {
		Fatalf("invalid symbol name: %s", name)
	}

	// Functions that can be inlined use numbered parameters so we can distingish them
	// from other names in their context after inlining (i.e., the parameter numbering
	// is a form of parameter rewriting). See issue 4326 for an example and test case.
	if forceObjFileStability || numbered {
		if !strings.Contains(name, "·") && f.Nname != nil && f.Nname.Name != nil && f.Nname.Name.Vargen > 0 {
			name = fmt.Sprintf("%s·%d", name, f.Nname.Name.Vargen) // append Vargen
		}
	} else {
		if i := strings.Index(name, "·"); i > 0 {
			name = name[:i] // cut off Vargen
		}
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
		if Minintval[TINT64].Cmp(x) <= 0 && x.Cmp(Maxintval[TINT64]) <= 0 {
			// common case: x fits into an int64 - use compact encoding
			p.tag(int64Tag)
			p.int64(x.Int64())
			return
		}
		// uncommon case: large x - use float encoding
		// (powers of 2 will be encoded efficiently with exponent)
		f := newMpflt()
		f.SetInt(x)
		p.tag(floatTag)
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

	case *NilVal:
		// not a constant but used in exported function bodies
		p.tag(nilTag)

	default:
		Fatalf("exporter: unexpected value %v (%T)", x, x)
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
		Fatalf("exporter: internal error")
	}

	p.int(sign)
	p.int(exp)
	p.string(string(mant.Bytes()))
}

// ----------------------------------------------------------------------------
// Inlined function bodies

// Approach: More or less closely follow what fmt.go is doing for FExp mode
// but instead of emitting the information textually, emit the node tree in
// binary form.

// TODO(gri) Improve tracing output. The current format is difficult to read.

// stmtList may emit more (or fewer) than len(list) nodes.
func (p *exporter) stmtList(list Nodes) {
	if p.trace {
		if list.Len() == 0 {
			p.tracef("{}")
		} else {
			p.tracef("{>")
			defer p.tracef("<\n}")
		}
	}

	for _, n := range list.Slice() {
		if p.trace {
			p.tracef("\n")
		}
		// TODO inlining produces expressions with ninits. we can't export these yet.
		// (from fmt.go:1461ff)
		if opprec[n.Op] < 0 {
			p.stmt(n)
		} else {
			p.expr(n)
		}
	}

	p.op(OEND)
}

func (p *exporter) exprList(list Nodes) {
	if p.trace {
		if list.Len() == 0 {
			p.tracef("{}")
		} else {
			p.tracef("{>")
			defer p.tracef("<\n}")
		}
	}

	for _, n := range list.Slice() {
		if p.trace {
			p.tracef("\n")
		}
		p.expr(n)
	}

	p.op(OEND)
}

func (p *exporter) elemList(list Nodes) {
	if p.trace {
		p.tracef("[ ")
	}
	p.int(list.Len())
	if p.trace {
		if list.Len() == 0 {
			p.tracef("] {}")
		} else {
			p.tracef("] {>")
			defer p.tracef("<\n}")
		}
	}

	for _, n := range list.Slice() {
		if p.trace {
			p.tracef("\n")
		}
		p.fieldSym(n.Left.Sym, false)
		p.expr(n.Right)
	}
}

func (p *exporter) expr(n *Node) {
	if p.trace {
		p.tracef("( ")
		defer p.tracef(") ")
	}

	// from nodefmt (fmt.go)
	//
	// nodefmt reverts nodes back to their original - we don't need to do
	// it because we are not bound to produce valid Go syntax when exporting
	//
	// if (fmtmode != FExp || n.Op != OLITERAL) && n.Orig != nil {
	// 	n = n.Orig
	// }

	// from exprfmt (fmt.go)
	for n != nil && n.Implicit && (n.Op == OIND || n.Op == OADDR) {
		n = n.Left
	}

	switch op := n.Op; op {
	// expressions
	// (somewhat closely following the structure of exprfmt in fmt.go)
	case OPAREN:
		p.expr(n.Left) // unparen

	// case ODDDARG:
	//	unimplemented - handled by default case

	// case OREGISTER:
	//	unimplemented - handled by default case

	case OLITERAL:
		if n.Val().Ctype() == CTNIL && n.Orig != nil && n.Orig != n {
			p.expr(n.Orig)
			break
		}
		p.op(OLITERAL)
		p.typ(unidealType(n.Type, n.Val()))
		p.value(n.Val())

	case ONAME:
		// Special case: name used as local variable in export.
		// _ becomes ~b%d internally; print as _ for export
		if n.Sym != nil && n.Sym.Name[0] == '~' && n.Sym.Name[1] == 'b' {
			p.op(ONAME)
			p.string("_") // inlined and customized version of p.sym(n)
			break
		}

		if n.Sym != nil && !isblank(n) && n.Name.Vargen > 0 {
			p.op(ONAME)
			p.sym(n)
			break
		}

		// Special case: explicit name of func (*T) method(...) is turned into pkg.(*T).method,
		// but for export, this should be rendered as (*pkg.T).meth.
		// These nodes have the special property that they are names with a left OTYPE and a right ONAME.
		if n.Left != nil && n.Left.Op == OTYPE && n.Right != nil && n.Right.Op == ONAME {
			p.op(OXDOT)
			p.expr(n.Left) // n.Left.Op == OTYPE
			p.fieldSym(n.Right.Sym, true)
			break
		}

		p.op(ONAME)
		p.sym(n)

	// case OPACK, ONONAME:
	// 	should have been resolved by typechecking - handled by default case

	case OTYPE:
		p.op(OTYPE)
		if p.bool(n.Type == nil) {
			p.sym(n)
		} else {
			p.typ(n.Type)
		}

	// case OTARRAY, OTMAP, OTCHAN, OTSTRUCT, OTINTER, OTFUNC:
	// 	should have been resolved by typechecking - handled by default case

	// case OCLOSURE:
	//	unimplemented - handled by default case

	// case OCOMPLIT:
	// 	should have been resolved by typechecking - handled by default case

	case OPTRLIT:
		p.op(OPTRLIT)
		p.expr(n.Left)
		p.bool(n.Implicit)

	case OSTRUCTLIT:
		p.op(OSTRUCTLIT)
		p.typ(n.Type)
		p.elemList(n.List) // special handling of field names

	case OARRAYLIT, OMAPLIT:
		p.op(OCOMPLIT)
		p.typ(n.Type)
		p.exprList(n.List)

	case OKEY:
		p.op(OKEY)
		p.exprsOrNil(n.Left, n.Right)

	// case OCALLPART:
	//	unimplemented - handled by default case

	case OXDOT, ODOT, ODOTPTR, ODOTINTER, ODOTMETH:
		p.op(OXDOT)
		p.expr(n.Left)
		p.fieldSym(n.Sym, true)

	case ODOTTYPE, ODOTTYPE2:
		p.op(ODOTTYPE)
		p.expr(n.Left)
		if p.bool(n.Right != nil) {
			p.expr(n.Right)
		} else {
			p.typ(n.Type)
		}

	case OINDEX, OINDEXMAP:
		p.op(OINDEX)
		p.expr(n.Left)
		p.expr(n.Right)

	case OSLICE, OSLICESTR, OSLICEARR:
		p.op(OSLICE)
		p.expr(n.Left)
		low, high, _ := n.SliceBounds()
		p.exprsOrNil(low, high)

	case OSLICE3, OSLICE3ARR:
		p.op(OSLICE3)
		p.expr(n.Left)
		low, high, max := n.SliceBounds()
		p.exprsOrNil(low, high)
		p.expr(max)

	case OCOPY, OCOMPLEX:
		// treated like other builtin calls (see e.g., OREAL)
		p.op(op)
		p.expr(n.Left)
		p.expr(n.Right)
		p.op(OEND)

	case OCONV, OCONVIFACE, OCONVNOP, OARRAYBYTESTR, OARRAYRUNESTR, OSTRARRAYBYTE, OSTRARRAYRUNE, ORUNESTR:
		p.op(OCONV)
		p.typ(n.Type)
		if n.Left != nil {
			p.expr(n.Left)
			p.op(OEND)
		} else {
			p.exprList(n.List) // emits terminating OEND
		}

	case OREAL, OIMAG, OAPPEND, OCAP, OCLOSE, ODELETE, OLEN, OMAKE, ONEW, OPANIC, ORECOVER, OPRINT, OPRINTN:
		p.op(op)
		if n.Left != nil {
			p.expr(n.Left)
			p.op(OEND)
		} else {
			p.exprList(n.List) // emits terminating OEND
		}
		// only append() calls may contain '...' arguments
		if op == OAPPEND {
			p.bool(n.Isddd)
		} else if n.Isddd {
			Fatalf("exporter: unexpected '...' with %s call", opnames[op])
		}

	case OCALL, OCALLFUNC, OCALLMETH, OCALLINTER, OGETG:
		p.op(OCALL)
		p.expr(n.Left)
		p.exprList(n.List)
		p.bool(n.Isddd)

	case OMAKEMAP, OMAKECHAN, OMAKESLICE:
		p.op(op) // must keep separate from OMAKE for importer
		p.typ(n.Type)
		switch {
		default:
			// empty list
			p.op(OEND)
		case n.List.Len() != 0: // pre-typecheck
			p.exprList(n.List) // emits terminating OEND
		case n.Right != nil:
			p.expr(n.Left)
			p.expr(n.Right)
			p.op(OEND)
		case n.Left != nil && (n.Op == OMAKESLICE || !n.Left.Type.IsUntyped()):
			p.expr(n.Left)
			p.op(OEND)
		}

	// unary expressions
	case OPLUS, OMINUS, OADDR, OCOM, OIND, ONOT, ORECV:
		p.op(op)
		p.expr(n.Left)

	// binary expressions
	case OADD, OAND, OANDAND, OANDNOT, ODIV, OEQ, OGE, OGT, OLE, OLT,
		OLSH, OMOD, OMUL, ONE, OOR, OOROR, ORSH, OSEND, OSUB, OXOR:
		p.op(op)
		p.expr(n.Left)
		p.expr(n.Right)

	case OADDSTR:
		p.op(OADDSTR)
		p.exprList(n.List)

	case OCMPSTR, OCMPIFACE:
		p.op(Op(n.Etype))
		p.expr(n.Left)
		p.expr(n.Right)

	case ODCLCONST:
		// if exporting, DCLCONST should just be removed as its usage
		// has already been replaced with literals
		// TODO(gri) these should not be exported in the first place
		// TODO(gri) why is this considered an expression in fmt.go?
		p.op(ODCLCONST)

	default:
		Fatalf("cannot export %s (%d) node\n"+
			"==> please file an issue and assign to gri@\n", n.Op, n.Op)
	}
}

// Caution: stmt will emit more than one node for statement nodes n that have a non-empty
// n.Ninit and where n cannot have a natural init section (such as in "if", "for", etc.).
func (p *exporter) stmt(n *Node) {
	if p.trace {
		p.tracef("( ")
		defer p.tracef(") ")
	}

	if n.Ninit.Len() > 0 && !stmtwithinit(n.Op) {
		if p.trace {
			p.tracef("( /* Ninits */ ")
		}

		// can't use stmtList here since we don't want the final OEND
		for _, n := range n.Ninit.Slice() {
			p.stmt(n)
		}

		if p.trace {
			p.tracef(") ")
		}
	}

	switch op := n.Op; op {
	case ODCL:
		p.op(ODCL)
		switch n.Left.Class {
		case PPARAM, PPARAMOUT, PAUTO, PAUTOHEAP:
			// TODO(gri) when is this not PAUTO?
			// Also, originally this didn't look like
			// the default case. Investigate.
			fallthrough
		default:
			// TODO(gri) Can we ever reach here?
			p.bool(false)
			p.sym(n.Left)
		}
		p.typ(n.Left.Type)

	// case ODCLFIELD:
	//	unimplemented - handled by default case

	case OAS, OASWB:
		// Don't export "v = <N>" initializing statements, hope they're always
		// preceded by the DCL which will be re-parsed and typecheck to reproduce
		// the "v = <N>" again.
		if n.Right != nil {
			p.op(OAS)
			p.expr(n.Left)
			p.expr(n.Right)
		}

	case OASOP:
		p.op(OASOP)
		p.int(int(n.Etype))
		p.expr(n.Left)
		if p.bool(!n.Implicit) {
			p.expr(n.Right)
		}

	case OAS2, OAS2DOTTYPE, OAS2FUNC, OAS2MAPR, OAS2RECV:
		p.op(OAS2)
		p.exprList(n.List)
		p.exprList(n.Rlist)

	case ORETURN:
		p.op(ORETURN)
		p.exprList(n.List)

	// case ORETJMP:
	// 	unreachable - generated by compiler for trampolin routines

	case OPROC, ODEFER:
		p.op(op)
		p.expr(n.Left)

	case OIF:
		p.op(OIF)
		p.stmtList(n.Ninit)
		p.expr(n.Left)
		p.stmtList(n.Nbody)
		p.stmtList(n.Rlist)

	case OFOR:
		p.op(OFOR)
		p.stmtList(n.Ninit)
		p.exprsOrNil(n.Left, n.Right)
		p.stmtList(n.Nbody)

	case ORANGE:
		p.op(ORANGE)
		p.stmtList(n.List)
		p.expr(n.Right)
		p.stmtList(n.Nbody)

	case OSELECT, OSWITCH:
		p.op(op)
		p.stmtList(n.Ninit)
		p.exprsOrNil(n.Left, nil)
		p.stmtList(n.List)

	case OCASE, OXCASE:
		p.op(OXCASE)
		p.stmtList(n.List)
		p.stmtList(n.Nbody)

	case OFALL, OXFALL:
		p.op(OXFALL)

	case OBREAK, OCONTINUE:
		p.op(op)
		p.exprsOrNil(n.Left, nil)

	case OEMPTY:
		// nothing to emit

	case OGOTO, OLABEL:
		p.op(op)
		p.expr(n.Left)

	default:
		Fatalf("exporter: CANNOT EXPORT: %s\nPlease notify gri@\n", n.Op)
	}
}

func (p *exporter) exprsOrNil(a, b *Node) {
	ab := 0
	if a != nil {
		ab |= 1
	}
	if b != nil {
		ab |= 2
	}
	p.int(ab)
	if ab&1 != 0 {
		p.expr(a)
	}
	if ab&2 != 0 {
		p.expr(b)
	}
}

func (p *exporter) fieldSym(s *Sym, short bool) {
	name := s.Name

	// remove leading "type." in method names ("(T).m" -> "m")
	if short {
		if i := strings.LastIndex(name, "."); i >= 0 {
			name = name[i+1:]
		}
	}

	// we should never see a _ (blank) here - these are accessible ("read") fields
	// TODO(gri) can we assert this with an explicit check?
	p.string(name)
	if !exportname(name) {
		p.pkg(s.Pkg)
	}
}

// sym must encode the _ (blank) identifier as a single string "_" since
// encoding for some nodes is based on this assumption (e.g. ONAME nodes).
func (p *exporter) sym(n *Node) {
	s := n.Sym
	if s.Pkg != nil {
		if len(s.Name) > 0 && s.Name[0] == '.' {
			Fatalf("exporter: exporting synthetic symbol %s", s.Name)
		}
	}

	if p.trace {
		p.tracef("{ SYM ")
		defer p.tracef("} ")
	}

	name := s.Name

	// remove leading "type." in method names ("(T).m" -> "m")
	if i := strings.LastIndex(name, "."); i >= 0 {
		name = name[i+1:]
	}

	if strings.Contains(name, "·") && n.Name.Vargen > 0 {
		Fatalf("exporter: unexpected · in symbol name")
	}

	if i := n.Name.Vargen; i > 0 {
		name = fmt.Sprintf("%s·%d", name, i)
	}

	p.string(name)
	if name != "_" {
		p.pkg(s.Pkg)
	}
}

func (p *exporter) bool(b bool) bool {
	if p.trace {
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

func (p *exporter) op(op Op) {
	if p.trace {
		p.tracef("[")
		defer p.tracef("= %s] ", op)
	}

	p.int(int(op))
}

// ----------------------------------------------------------------------------
// Low-level encoders

func (p *exporter) index(marker byte, index int) {
	if index < 0 {
		Fatalf("exporter: invalid index < 0")
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
		Fatalf("exporter: invalid tag >= 0")
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
// it easy for a reader to detect if it is "out of sync". Used only
// if debugFormat is set.
func (p *exporter) marker(m byte) {
	p.rawByte(m)
	// Uncomment this for help tracking down the location
	// of an incorrect marker when running in debugFormat.
	// if p.trace {
	// 	p.tracef("#%d ", p.written)
	// }
	p.rawInt64(int64(p.written))
}

// rawInt64 should only be used by low-level encoders
func (p *exporter) rawInt64(x int64) {
	var tmp [binary.MaxVarintLen64]byte
	n := binary.PutVarint(tmp[:], x)
	for i := 0; i < n; i++ {
		p.rawByte(tmp[i])
	}
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

// ----------------------------------------------------------------------------
// Export format

// Tags. Must be < 0.
const (
	// Objects
	packageTag = -(iota + 1)
	constTag
	typeTag
	varTag
	funcTag
	endTag

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
	nilTag
	unknownTag // not used by gc (only appears in packages with errors)
)

// Debugging support.
// (tagString is only used when tracing is enabled)
var tagString = [...]string{
	// Objects
	-packageTag: "package",
	-constTag:   "const",
	-typeTag:    "type",
	-varTag:     "var",
	-funcTag:    "func",
	-endTag:     "end",

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
	-nilTag:      "nil",
	-unknownTag:  "unknown",
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
	Fatalf("exporter: unknown Ctype")
	return nil
}

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

			// invalid type (package contains errors)
			Types[Txxx],

			// any type, for builtin export data
			Types[TANY],
		}
	}
	return predecl
}

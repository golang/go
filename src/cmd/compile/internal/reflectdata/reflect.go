// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package reflectdata

import (
	"encoding/binary"
	"fmt"
	"internal/abi"
	"internal/buildcfg"
	"slices"
	"sort"
	"strings"
	"sync"

	"cmd/compile/internal/base"
	"cmd/compile/internal/bitvec"
	"cmd/compile/internal/compare"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/objw"
	"cmd/compile/internal/rttype"
	"cmd/compile/internal/staticdata"
	"cmd/compile/internal/typebits"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/objabi"
	"cmd/internal/src"
)

type ptabEntry struct {
	s *types.Sym
	t *types.Type
}

// runtime interface and reflection data structures
var (
	// protects signatset and signatslice
	signatmu sync.Mutex
	// Tracking which types need runtime type descriptor
	signatset = make(map[*types.Type]struct{})
	// Queue of types wait to be generated runtime type descriptor
	signatslice []typeAndStr

	gcsymmu  sync.Mutex // protects gcsymset and gcsymslice
	gcsymset = make(map[*types.Type]struct{})
)

type typeSig struct {
	name  *types.Sym
	isym  *obj.LSym
	tsym  *obj.LSym
	type_ *types.Type
	mtype *types.Type
}

func commonSize() int { return int(rttype.Type.Size()) } // Sizeof(runtime._type{})

func uncommonSize(t *types.Type) int { // Sizeof(runtime.uncommontype{})
	if t.Sym() == nil && len(methods(t)) == 0 {
		return 0
	}
	return int(rttype.UncommonType.Size())
}

func makefield(name string, t *types.Type) *types.Field {
	sym := (*types.Pkg)(nil).Lookup(name)
	return types.NewField(src.NoXPos, sym, t)
}

// methods returns the methods of the non-interface type t, sorted by name.
// Generates stub functions as needed.
func methods(t *types.Type) []*typeSig {
	if t.HasShape() {
		// Shape types have no methods.
		return nil
	}
	// method type
	mt := types.ReceiverBaseType(t)

	if mt == nil {
		return nil
	}
	typecheck.CalcMethods(mt)

	// make list of methods for t,
	// generating code if necessary.
	var ms []*typeSig
	for _, f := range mt.AllMethods() {
		if f.Sym == nil {
			base.Fatalf("method with no sym on %v", mt)
		}
		if !f.IsMethod() {
			base.Fatalf("non-method on %v method %v %v", mt, f.Sym, f)
		}
		if f.Type.Recv() == nil {
			base.Fatalf("receiver with no type on %v method %v %v", mt, f.Sym, f)
		}
		if f.Nointerface() && !t.IsFullyInstantiated() {
			// Skip creating method wrappers if f is nointerface. But, if
			// t is an instantiated type, we still have to call
			// methodWrapper, because methodWrapper generates the actual
			// generic method on the type as well.
			continue
		}

		// get receiver type for this particular method.
		// if pointer receiver but non-pointer t and
		// this is not an embedded pointer inside a struct,
		// method does not apply.
		if !types.IsMethodApplicable(t, f) {
			continue
		}

		sig := &typeSig{
			name:  f.Sym,
			isym:  methodWrapper(t, f, true),
			tsym:  methodWrapper(t, f, false),
			type_: typecheck.NewMethodType(f.Type, t),
			mtype: typecheck.NewMethodType(f.Type, nil),
		}
		if f.Nointerface() {
			// In the case of a nointerface method on an instantiated
			// type, don't actually append the typeSig.
			continue
		}
		ms = append(ms, sig)
	}

	return ms
}

// imethods returns the methods of the interface type t, sorted by name.
func imethods(t *types.Type) []*typeSig {
	var methods []*typeSig
	for _, f := range t.AllMethods() {
		if f.Type.Kind() != types.TFUNC || f.Sym == nil {
			continue
		}
		if f.Sym.IsBlank() {
			base.Fatalf("unexpected blank symbol in interface method set")
		}
		if n := len(methods); n > 0 {
			last := methods[n-1]
			if types.CompareSyms(last.name, f.Sym) >= 0 {
				base.Fatalf("sigcmp vs sortinter %v %v", last.name, f.Sym)
			}
		}

		sig := &typeSig{
			name:  f.Sym,
			mtype: f.Type,
			type_: typecheck.NewMethodType(f.Type, nil),
		}
		methods = append(methods, sig)

		// NOTE(rsc): Perhaps an oversight that
		// IfaceType.Method is not in the reflect data.
		// Generate the method body, so that compiled
		// code can refer to it.
		methodWrapper(t, f, false)
	}

	return methods
}

func dimportpath(p *types.Pkg) {
	if p.Pathsym != nil {
		return
	}

	if p == types.LocalPkg && base.Ctxt.Pkgpath == "" {
		panic("missing pkgpath")
	}

	// If we are compiling the runtime package, there are two runtime packages around
	// -- localpkg and Pkgs.Runtime. We don't want to produce import path symbols for
	// both of them, so just produce one for localpkg.
	if base.Ctxt.Pkgpath == "runtime" && p == ir.Pkgs.Runtime {
		return
	}

	s := base.Ctxt.Lookup("type:.importpath." + p.Prefix + ".")
	ot := dnameData(s, 0, p.Path, "", nil, false, false)
	objw.Global(s, int32(ot), obj.DUPOK|obj.RODATA)
	s.Set(obj.AttrContentAddressable, true)
	p.Pathsym = s
}

func dgopkgpath(c rttype.Cursor, pkg *types.Pkg) {
	c = c.Field("Bytes")
	if pkg == nil {
		c.WritePtr(nil)
		return
	}

	dimportpath(pkg)
	c.WritePtr(pkg.Pathsym)
}

// dgopkgpathOff writes an offset relocation to the pkg path symbol to c.
func dgopkgpathOff(c rttype.Cursor, pkg *types.Pkg) {
	if pkg == nil {
		c.WriteInt32(0)
		return
	}

	dimportpath(pkg)
	c.WriteSymPtrOff(pkg.Pathsym, false)
}

// dnameField dumps a reflect.name for a struct field.
func dnameField(c rttype.Cursor, spkg *types.Pkg, ft *types.Field) {
	if !types.IsExported(ft.Sym.Name) && ft.Sym.Pkg != spkg {
		base.Fatalf("package mismatch for %v", ft.Sym)
	}
	nsym := dname(ft.Sym.Name, ft.Note, nil, types.IsExported(ft.Sym.Name), ft.Embedded != 0)
	c.Field("Bytes").WritePtr(nsym)
}

// dnameData writes the contents of a reflect.name into s at offset ot.
func dnameData(s *obj.LSym, ot int, name, tag string, pkg *types.Pkg, exported, embedded bool) int {
	if len(name) >= 1<<29 {
		base.Fatalf("name too long: %d %s...", len(name), name[:1024])
	}
	if len(tag) >= 1<<29 {
		base.Fatalf("tag too long: %d %s...", len(tag), tag[:1024])
	}
	var nameLen [binary.MaxVarintLen64]byte
	nameLenLen := binary.PutUvarint(nameLen[:], uint64(len(name)))
	var tagLen [binary.MaxVarintLen64]byte
	tagLenLen := binary.PutUvarint(tagLen[:], uint64(len(tag)))

	// Encode name and tag. See reflect/type.go for details.
	var bits byte
	l := 1 + nameLenLen + len(name)
	if exported {
		bits |= 1 << 0
	}
	if len(tag) > 0 {
		l += tagLenLen + len(tag)
		bits |= 1 << 1
	}
	if pkg != nil {
		bits |= 1 << 2
	}
	if embedded {
		bits |= 1 << 3
	}
	b := make([]byte, l)
	b[0] = bits
	copy(b[1:], nameLen[:nameLenLen])
	copy(b[1+nameLenLen:], name)
	if len(tag) > 0 {
		tb := b[1+nameLenLen+len(name):]
		copy(tb, tagLen[:tagLenLen])
		copy(tb[tagLenLen:], tag)
	}

	ot = int(s.WriteBytes(base.Ctxt, int64(ot), b))

	if pkg != nil {
		c := rttype.NewCursor(s, int64(ot), types.Types[types.TUINT32])
		dgopkgpathOff(c, pkg)
		ot += 4
	}

	return ot
}

var dnameCount int

// dname creates a reflect.name for a struct field or method.
func dname(name, tag string, pkg *types.Pkg, exported, embedded bool) *obj.LSym {
	// Write out data as "type:." to signal two things to the
	// linker, first that when dynamically linking, the symbol
	// should be moved to a relro section, and second that the
	// contents should not be decoded as a type.
	sname := "type:.namedata."
	if pkg == nil {
		// In the common case, share data with other packages.
		if name == "" {
			if exported {
				sname += "-noname-exported." + tag
			} else {
				sname += "-noname-unexported." + tag
			}
		} else {
			if exported {
				sname += name + "." + tag
			} else {
				sname += name + "-" + tag
			}
		}
	} else {
		// TODO(mdempsky): We should be able to share these too (except
		// maybe when dynamic linking).
		sname = fmt.Sprintf("%s%s.%d", sname, types.LocalPkg.Prefix, dnameCount)
		dnameCount++
	}
	if embedded {
		sname += ".embedded"
	}
	s := base.Ctxt.Lookup(sname)
	if len(s.P) > 0 {
		return s
	}
	ot := dnameData(s, 0, name, tag, pkg, exported, embedded)
	objw.Global(s, int32(ot), obj.DUPOK|obj.RODATA)
	s.Set(obj.AttrContentAddressable, true)
	return s
}

// dextratype dumps the fields of a runtime.uncommontype.
// dataAdd is the offset in bytes after the header where the
// backing array of the []method field should be written.
func dextratype(lsym *obj.LSym, off int64, t *types.Type, dataAdd int) {
	m := methods(t)
	if t.Sym() == nil && len(m) == 0 {
		base.Fatalf("extra requested of type with no extra info %v", t)
	}
	noff := types.RoundUp(off, int64(types.PtrSize))
	if noff != off {
		base.Fatalf("unexpected alignment in dextratype for %v", t)
	}

	for _, a := range m {
		writeType(a.type_)
	}

	c := rttype.NewCursor(lsym, off, rttype.UncommonType)
	dgopkgpathOff(c.Field("PkgPath"), typePkg(t))

	dataAdd += uncommonSize(t)
	mcount := len(m)
	if mcount != int(uint16(mcount)) {
		base.Fatalf("too many methods on %v: %d", t, mcount)
	}
	xcount := sort.Search(mcount, func(i int) bool { return !types.IsExported(m[i].name.Name) })
	if dataAdd != int(uint32(dataAdd)) {
		base.Fatalf("methods are too far away on %v: %d", t, dataAdd)
	}

	c.Field("Mcount").WriteUint16(uint16(mcount))
	c.Field("Xcount").WriteUint16(uint16(xcount))
	c.Field("Moff").WriteUint32(uint32(dataAdd))
	// Note: there is an unused uint32 field here.

	// Write the backing array for the []method field.
	array := rttype.NewArrayCursor(lsym, off+int64(dataAdd), rttype.Method, mcount)
	for i, a := range m {
		exported := types.IsExported(a.name.Name)
		var pkg *types.Pkg
		if !exported && a.name.Pkg != typePkg(t) {
			pkg = a.name.Pkg
		}
		nsym := dname(a.name.Name, "", pkg, exported, false)

		e := array.Elem(i)
		e.Field("Name").WriteSymPtrOff(nsym, false)
		dmethodptrOff(e.Field("Mtyp"), writeType(a.mtype))
		dmethodptrOff(e.Field("Ifn"), a.isym)
		dmethodptrOff(e.Field("Tfn"), a.tsym)
	}
}

func typePkg(t *types.Type) *types.Pkg {
	tsym := t.Sym()
	if tsym == nil {
		switch t.Kind() {
		case types.TARRAY, types.TSLICE, types.TPTR, types.TCHAN:
			if t.Elem() != nil {
				tsym = t.Elem().Sym()
			}
		}
	}
	if tsym != nil && tsym.Pkg != types.BuiltinPkg {
		return tsym.Pkg
	}
	return nil
}

func dmethodptrOff(c rttype.Cursor, x *obj.LSym) {
	c.WriteInt32(0)
	c.Reloc(obj.Reloc{Type: objabi.R_METHODOFF, Sym: x})
}

var kinds = []abi.Kind{
	types.TINT:        abi.Int,
	types.TUINT:       abi.Uint,
	types.TINT8:       abi.Int8,
	types.TUINT8:      abi.Uint8,
	types.TINT16:      abi.Int16,
	types.TUINT16:     abi.Uint16,
	types.TINT32:      abi.Int32,
	types.TUINT32:     abi.Uint32,
	types.TINT64:      abi.Int64,
	types.TUINT64:     abi.Uint64,
	types.TUINTPTR:    abi.Uintptr,
	types.TFLOAT32:    abi.Float32,
	types.TFLOAT64:    abi.Float64,
	types.TBOOL:       abi.Bool,
	types.TSTRING:     abi.String,
	types.TPTR:        abi.Pointer,
	types.TSTRUCT:     abi.Struct,
	types.TINTER:      abi.Interface,
	types.TCHAN:       abi.Chan,
	types.TMAP:        abi.Map,
	types.TARRAY:      abi.Array,
	types.TSLICE:      abi.Slice,
	types.TFUNC:       abi.Func,
	types.TCOMPLEX64:  abi.Complex64,
	types.TCOMPLEX128: abi.Complex128,
	types.TUNSAFEPTR:  abi.UnsafePointer,
}

var (
	memhashvarlen  *obj.LSym
	memequalvarlen *obj.LSym
)

// dcommontype dumps the contents of a reflect.rtype (runtime._type) to c.
func dcommontype(c rttype.Cursor, t *types.Type) {
	types.CalcSize(t)
	eqfunc := geneq(t)

	sptrWeak := true
	var sptr *obj.LSym
	if !t.IsPtr() || t.IsPtrElem() {
		tptr := types.NewPtr(t)
		if t.Sym() != nil || methods(tptr) != nil {
			sptrWeak = false
		}
		sptr = writeType(tptr)
	}

	gcsym, onDemand, ptrdata := dgcsym(t, true, true)
	if !onDemand {
		delete(gcsymset, t)
	}

	// ../../../../reflect/type.go:/^type.rtype
	// actual type structure
	//	type rtype struct {
	//		size          uintptr
	//		ptrdata       uintptr
	//		hash          uint32
	//		tflag         tflag
	//		align         uint8
	//		fieldAlign    uint8
	//		kind          uint8
	//		equal         func(unsafe.Pointer, unsafe.Pointer) bool
	//		gcdata        *byte
	//		str           nameOff
	//		ptrToThis     typeOff
	//	}
	c.Field("Size_").WriteUintptr(uint64(t.Size()))
	c.Field("PtrBytes").WriteUintptr(uint64(ptrdata))
	c.Field("Hash").WriteUint32(types.TypeHash(t))

	var tflag abi.TFlag
	if uncommonSize(t) != 0 {
		tflag |= abi.TFlagUncommon
	}
	if t.Sym() != nil && t.Sym().Name != "" {
		tflag |= abi.TFlagNamed
	}
	if compare.IsRegularMemory(t) {
		tflag |= abi.TFlagRegularMemory
	}
	if onDemand {
		tflag |= abi.TFlagGCMaskOnDemand
	}

	exported := false
	p := t.NameString()
	// If we're writing out type T,
	// we are very likely to write out type *T as well.
	// Use the string "*T"[1:] for "T", so that the two
	// share storage. This is a cheap way to reduce the
	// amount of space taken up by reflect strings.
	if !strings.HasPrefix(p, "*") {
		p = "*" + p
		tflag |= abi.TFlagExtraStar
		if t.Sym() != nil {
			exported = types.IsExported(t.Sym().Name)
		}
	} else {
		if t.Elem() != nil && t.Elem().Sym() != nil {
			exported = types.IsExported(t.Elem().Sym().Name)
		}
	}

	if tflag != abi.TFlag(uint8(tflag)) {
		// this should optimize away completely
		panic("Unexpected change in size of abi.TFlag")
	}
	c.Field("TFlag").WriteUint8(uint8(tflag))

	// runtime (and common sense) expects alignment to be a power of two.
	i := int(uint8(t.Alignment()))

	if i == 0 {
		i = 1
	}
	if i&(i-1) != 0 {
		base.Fatalf("invalid alignment %d for %v", uint8(t.Alignment()), t)
	}
	c.Field("Align_").WriteUint8(uint8(t.Alignment()))
	c.Field("FieldAlign_").WriteUint8(uint8(t.Alignment()))

	kind := kinds[t.Kind()]
	if types.IsDirectIface(t) {
		kind |= abi.KindDirectIface
	}
	c.Field("Kind_").WriteUint8(uint8(kind))

	c.Field("Equal").WritePtr(eqfunc)
	c.Field("GCData").WritePtr(gcsym)

	nsym := dname(p, "", nil, exported, false)
	c.Field("Str").WriteSymPtrOff(nsym, false)
	c.Field("PtrToThis").WriteSymPtrOff(sptr, sptrWeak)
}

// TrackSym returns the symbol for tracking use of field/method f, assumed
// to be a member of struct/interface type t.
func TrackSym(t *types.Type, f *types.Field) *obj.LSym {
	return base.PkgLinksym("go:track", t.LinkString()+"."+f.Sym.Name, obj.ABI0)
}

func TypeSymPrefix(prefix string, t *types.Type) *types.Sym {
	p := prefix + "." + t.LinkString()
	s := types.TypeSymLookup(p)

	// This function is for looking up type-related generated functions
	// (e.g. eq and hash). Make sure they are indeed generated.
	signatmu.Lock()
	NeedRuntimeType(t)
	signatmu.Unlock()

	//print("algsym: %s -> %+S\n", p, s);

	return s
}

func TypeSym(t *types.Type) *types.Sym {
	if t == nil || (t.IsPtr() && t.Elem() == nil) || t.IsUntyped() {
		base.Fatalf("TypeSym %v", t)
	}
	if t.Kind() == types.TFUNC && t.Recv() != nil {
		base.Fatalf("misuse of method type: %v", t)
	}
	s := types.TypeSym(t)
	signatmu.Lock()
	NeedRuntimeType(t)
	signatmu.Unlock()
	return s
}

func TypeLinksymPrefix(prefix string, t *types.Type) *obj.LSym {
	return TypeSymPrefix(prefix, t).Linksym()
}

func TypeLinksymLookup(name string) *obj.LSym {
	return types.TypeSymLookup(name).Linksym()
}

func TypeLinksym(t *types.Type) *obj.LSym {
	lsym := TypeSym(t).Linksym()
	signatmu.Lock()
	if lsym.Extra == nil {
		ti := lsym.NewTypeInfo()
		ti.Type = t
	}
	signatmu.Unlock()
	return lsym
}

// TypePtrAt returns an expression that evaluates to the
// *runtime._type value for t.
func TypePtrAt(pos src.XPos, t *types.Type) *ir.AddrExpr {
	return typecheck.LinksymAddr(pos, TypeLinksym(t), types.Types[types.TUINT8])
}

// ITabLsym returns the LSym representing the itab for concrete type typ implementing
// interface iface. A dummy tab will be created in the unusual case where typ doesn't
// implement iface. Normally, this wouldn't happen, because the typechecker would
// have reported a compile-time error. This situation can only happen when the
// destination type of a type assert or a type in a type switch is parameterized, so
// it may sometimes, but not always, be a type that can't implement the specified
// interface.
func ITabLsym(typ, iface *types.Type) *obj.LSym {
	s, existed := ir.Pkgs.Itab.LookupOK(typ.LinkString() + "," + iface.LinkString())
	lsym := s.Linksym()

	if !existed {
		writeITab(lsym, typ, iface, true)
	}
	return lsym
}

// ITabAddrAt returns an expression that evaluates to the
// *runtime.itab value for concrete type typ implementing interface
// iface.
func ITabAddrAt(pos src.XPos, typ, iface *types.Type) *ir.AddrExpr {
	s, existed := ir.Pkgs.Itab.LookupOK(typ.LinkString() + "," + iface.LinkString())
	lsym := s.Linksym()

	if !existed {
		writeITab(lsym, typ, iface, false)
	}

	return typecheck.LinksymAddr(pos, lsym, types.Types[types.TUINT8])
}

// needkeyupdate reports whether map updates with t as a key
// need the key to be updated.
func needkeyupdate(t *types.Type) bool {
	switch t.Kind() {
	case types.TBOOL, types.TINT, types.TUINT, types.TINT8, types.TUINT8, types.TINT16, types.TUINT16, types.TINT32, types.TUINT32,
		types.TINT64, types.TUINT64, types.TUINTPTR, types.TPTR, types.TUNSAFEPTR, types.TCHAN:
		return false

	case types.TFLOAT32, types.TFLOAT64, types.TCOMPLEX64, types.TCOMPLEX128, // floats and complex can be +0/-0
		types.TINTER,
		types.TSTRING: // strings might have smaller backing stores
		return true

	case types.TARRAY:
		return needkeyupdate(t.Elem())

	case types.TSTRUCT:
		for _, t1 := range t.Fields() {
			if needkeyupdate(t1.Type) {
				return true
			}
		}
		return false

	default:
		base.Fatalf("bad type for map key: %v", t)
		return true
	}
}

// hashMightPanic reports whether the hash of a map key of type t might panic.
func hashMightPanic(t *types.Type) bool {
	switch t.Kind() {
	case types.TINTER:
		return true

	case types.TARRAY:
		return hashMightPanic(t.Elem())

	case types.TSTRUCT:
		for _, t1 := range t.Fields() {
			if hashMightPanic(t1.Type) {
				return true
			}
		}
		return false

	default:
		return false
	}
}

// formalType replaces predeclared aliases with real types.
// They've been separate internally to make error messages
// better, but we have to merge them in the reflect tables.
func formalType(t *types.Type) *types.Type {
	switch t {
	case types.AnyType, types.ByteType, types.RuneType:
		return types.Types[t.Kind()]
	}
	return t
}

func writeType(t *types.Type) *obj.LSym {
	t = formalType(t)
	if t.IsUntyped() {
		base.Fatalf("writeType %v", t)
	}

	s := types.TypeSym(t)
	lsym := s.Linksym()

	// special case (look for runtime below):
	// when compiling package runtime,
	// emit the type structures for int, float, etc.
	tbase := t
	if t.IsPtr() && t.Sym() == nil && t.Elem().Sym() != nil {
		tbase = t.Elem()
	}
	if tbase.Kind() == types.TFORW {
		base.Fatalf("unresolved defined type: %v", tbase)
	}

	// This is a fake type we generated for our builtin pseudo-runtime
	// package. We'll emit a description for the real type while
	// compiling package runtime, so we don't need or want to emit one
	// from this fake type.
	if sym := tbase.Sym(); sym != nil && sym.Pkg == ir.Pkgs.Runtime {
		return lsym
	}

	if s.Siggen() {
		return lsym
	}
	s.SetSiggen(true)

	if !NeedEmit(tbase) {
		if i := typecheck.BaseTypeIndex(t); i >= 0 {
			lsym.Pkg = tbase.Sym().Pkg.Prefix
			lsym.SymIdx = int32(i)
			lsym.Set(obj.AttrIndexed, true)
		}

		// TODO(mdempsky): Investigate whether this still happens.
		// If we know we don't need to emit code for a type,
		// we should have a link-symbol index for it.
		// See also TODO in NeedEmit.
		return lsym
	}

	// Type layout                          Written by               Marker
	// +--------------------------------+                            - 0
	// | abi/internal.Type              |   dcommontype
	// +--------------------------------+                            - A
	// | additional type-dependent      |   code in the switch below
	// | fields, e.g.                   |
	// | abi/internal.ArrayType.Len     |
	// +--------------------------------+                            - B
	// | internal/abi.UncommonType      |   dextratype
	// | This section is optional,      |
	// | if type has a name or methods  |
	// +--------------------------------+                            - C
	// | variable-length data           |   code in the switch below
	// | referenced by                  |
	// | type-dependent fields, e.g.    |
	// | abi/internal.StructType.Fields |
	// | dataAdd = size of this section |
	// +--------------------------------+                            - D
	// | method list, if any            |   dextratype
	// +--------------------------------+                            - E

	// UncommonType section is included if we have a name or a method.
	extra := t.Sym() != nil || len(methods(t)) != 0

	// Decide the underlying type of the descriptor, and remember
	// the size we need for variable-length data.
	var rt *types.Type
	dataAdd := 0
	switch t.Kind() {
	default:
		rt = rttype.Type
	case types.TARRAY:
		rt = rttype.ArrayType
	case types.TSLICE:
		rt = rttype.SliceType
	case types.TCHAN:
		rt = rttype.ChanType
	case types.TFUNC:
		rt = rttype.FuncType
		dataAdd = (t.NumRecvs() + t.NumParams() + t.NumResults()) * types.PtrSize
	case types.TINTER:
		rt = rttype.InterfaceType
		dataAdd = len(imethods(t)) * int(rttype.IMethod.Size())
	case types.TMAP:
		if buildcfg.Experiment.SwissMap {
			rt = rttype.SwissMapType
		} else {
			rt = rttype.OldMapType
		}
	case types.TPTR:
		rt = rttype.PtrType
		// TODO: use rttype.Type for Elem() is ANY?
	case types.TSTRUCT:
		rt = rttype.StructType
		dataAdd = t.NumFields() * int(rttype.StructField.Size())
	}

	// Compute offsets of each section.
	B := rt.Size()
	C := B
	if extra {
		C = B + rttype.UncommonType.Size()
	}
	D := C + int64(dataAdd)
	E := D + int64(len(methods(t)))*rttype.Method.Size()

	// Write the runtime._type
	c := rttype.NewCursor(lsym, 0, rt)
	if rt == rttype.Type {
		dcommontype(c, t)
	} else {
		dcommontype(c.Field("Type"), t)
	}

	// Write additional type-specific data
	// (Both the fixed size and variable-sized sections.)
	switch t.Kind() {
	case types.TARRAY:
		// internal/abi.ArrayType
		s1 := writeType(t.Elem())
		t2 := types.NewSlice(t.Elem())
		s2 := writeType(t2)
		c.Field("Elem").WritePtr(s1)
		c.Field("Slice").WritePtr(s2)
		c.Field("Len").WriteUintptr(uint64(t.NumElem()))

	case types.TSLICE:
		// internal/abi.SliceType
		s1 := writeType(t.Elem())
		c.Field("Elem").WritePtr(s1)

	case types.TCHAN:
		// internal/abi.ChanType
		s1 := writeType(t.Elem())
		c.Field("Elem").WritePtr(s1)
		c.Field("Dir").WriteInt(int64(t.ChanDir()))

	case types.TFUNC:
		// internal/abi.FuncType
		for _, t1 := range t.RecvParamsResults() {
			writeType(t1.Type)
		}
		inCount := t.NumRecvs() + t.NumParams()
		outCount := t.NumResults()
		if t.IsVariadic() {
			outCount |= 1 << 15
		}

		c.Field("InCount").WriteUint16(uint16(inCount))
		c.Field("OutCount").WriteUint16(uint16(outCount))

		// Array of rtype pointers follows funcType.
		typs := t.RecvParamsResults()
		array := rttype.NewArrayCursor(lsym, C, types.Types[types.TUNSAFEPTR], len(typs))
		for i, t1 := range typs {
			array.Elem(i).WritePtr(writeType(t1.Type))
		}

	case types.TINTER:
		// internal/abi.InterfaceType
		m := imethods(t)
		n := len(m)
		for _, a := range m {
			writeType(a.type_)
		}

		var tpkg *types.Pkg
		if t.Sym() != nil && t != types.Types[t.Kind()] && t != types.ErrorType {
			tpkg = t.Sym().Pkg
		}
		dgopkgpath(c.Field("PkgPath"), tpkg)
		c.Field("Methods").WriteSlice(lsym, C, int64(n), int64(n))

		array := rttype.NewArrayCursor(lsym, C, rttype.IMethod, n)
		for i, a := range m {
			exported := types.IsExported(a.name.Name)
			var pkg *types.Pkg
			if !exported && a.name.Pkg != tpkg {
				pkg = a.name.Pkg
			}
			nsym := dname(a.name.Name, "", pkg, exported, false)

			e := array.Elem(i)
			e.Field("Name").WriteSymPtrOff(nsym, false)
			e.Field("Typ").WriteSymPtrOff(writeType(a.type_), false)
		}

	case types.TMAP:
		if buildcfg.Experiment.SwissMap {
			writeSwissMapType(t, lsym, c)
		} else {
			writeOldMapType(t, lsym, c)
		}

	case types.TPTR:
		// internal/abi.PtrType
		if t.Elem().Kind() == types.TANY {
			base.Fatalf("bad pointer base type")
		}

		s1 := writeType(t.Elem())
		c.Field("Elem").WritePtr(s1)

	case types.TSTRUCT:
		// internal/abi.StructType
		fields := t.Fields()
		for _, t1 := range fields {
			writeType(t1.Type)
		}

		// All non-exported struct field names within a struct
		// type must originate from a single package. By
		// identifying and recording that package within the
		// struct type descriptor, we can omit that
		// information from the field descriptors.
		var spkg *types.Pkg
		for _, f := range fields {
			if !types.IsExported(f.Sym.Name) {
				spkg = f.Sym.Pkg
				break
			}
		}

		dgopkgpath(c.Field("PkgPath"), spkg)
		c.Field("Fields").WriteSlice(lsym, C, int64(len(fields)), int64(len(fields)))

		array := rttype.NewArrayCursor(lsym, C, rttype.StructField, len(fields))
		for i, f := range fields {
			e := array.Elem(i)
			dnameField(e.Field("Name"), spkg, f)
			e.Field("Typ").WritePtr(writeType(f.Type))
			e.Field("Offset").WriteUintptr(uint64(f.Offset))
		}
	}

	// Write the extra info, if any.
	if extra {
		dextratype(lsym, B, t, dataAdd)
	}

	// Note: DUPOK is required to ensure that we don't end up with more
	// than one type descriptor for a given type, if the type descriptor
	// can be defined in multiple packages, that is, unnamed types,
	// instantiated types and shape types.
	dupok := 0
	if tbase.Sym() == nil || tbase.IsFullyInstantiated() || tbase.HasShape() {
		dupok = obj.DUPOK
	}

	objw.Global(lsym, int32(E), int16(dupok|obj.RODATA))

	// The linker will leave a table of all the typelinks for
	// types in the binary, so the runtime can find them.
	//
	// When buildmode=shared, all types are in typelinks so the
	// runtime can deduplicate type pointers.
	keep := base.Ctxt.Flag_dynlink
	if !keep && t.Sym() == nil {
		// For an unnamed type, we only need the link if the type can
		// be created at run time by reflect.PointerTo and similar
		// functions. If the type exists in the program, those
		// functions must return the existing type structure rather
		// than creating a new one.
		switch t.Kind() {
		case types.TPTR, types.TARRAY, types.TCHAN, types.TFUNC, types.TMAP, types.TSLICE, types.TSTRUCT:
			keep = true
		}
	}
	// Do not put Noalg types in typelinks.  See issue #22605.
	if types.TypeHasNoAlg(t) {
		keep = false
	}
	lsym.Set(obj.AttrMakeTypelink, keep)

	return lsym
}

// InterfaceMethodOffset returns the offset of the i-th method in the interface
// type descriptor, ityp.
func InterfaceMethodOffset(ityp *types.Type, i int64) int64 {
	// interface type descriptor layout is struct {
	//   _type        // commonSize
	//   pkgpath      // 1 word
	//   []imethod    // 3 words (pointing to [...]imethod below)
	//   uncommontype // uncommonSize
	//   [...]imethod
	// }
	// The size of imethod is 8.
	return int64(commonSize()+4*types.PtrSize+uncommonSize(ityp)) + i*8
}

// NeedRuntimeType ensures that a runtime type descriptor is emitted for t.
func NeedRuntimeType(t *types.Type) {
	if _, ok := signatset[t]; !ok {
		signatset[t] = struct{}{}
		signatslice = append(signatslice, typeAndStr{t: t, short: types.TypeSymName(t), regular: t.String()})
	}
}

func WriteRuntimeTypes() {
	// Process signatslice. Use a loop, as writeType adds
	// entries to signatslice while it is being processed.
	for len(signatslice) > 0 {
		signats := signatslice
		// Sort for reproducible builds.
		slices.SortFunc(signats, typesStrCmp)
		for _, ts := range signats {
			t := ts.t
			writeType(t)
			if t.Sym() != nil {
				writeType(types.NewPtr(t))
			}
		}
		signatslice = signatslice[len(signats):]
	}
}

func WriteGCSymbols() {
	// Emit GC data symbols.
	gcsyms := make([]typeAndStr, 0, len(gcsymset))
	for t := range gcsymset {
		gcsyms = append(gcsyms, typeAndStr{t: t, short: types.TypeSymName(t), regular: t.String()})
	}
	slices.SortFunc(gcsyms, typesStrCmp)
	for _, ts := range gcsyms {
		dgcsym(ts.t, true, false)
	}
}

// writeITab writes the itab for concrete type typ implementing interface iface. If
// allowNonImplement is true, allow the case where typ does not implement iface, and just
// create a dummy itab with zeroed-out method entries.
func writeITab(lsym *obj.LSym, typ, iface *types.Type, allowNonImplement bool) {
	// TODO(mdempsky): Fix methodWrapper, geneq, and genhash (and maybe
	// others) to stop clobbering these.
	oldpos, oldfn := base.Pos, ir.CurFunc
	defer func() { base.Pos, ir.CurFunc = oldpos, oldfn }()

	if typ == nil || (typ.IsPtr() && typ.Elem() == nil) || typ.IsUntyped() || iface == nil || !iface.IsInterface() || iface.IsEmptyInterface() {
		base.Fatalf("writeITab(%v, %v)", typ, iface)
	}

	sigs := iface.AllMethods()
	entries := make([]*obj.LSym, 0, len(sigs))

	// both sigs and methods are sorted by name,
	// so we can find the intersection in a single pass
	for _, m := range methods(typ) {
		if m.name == sigs[0].Sym {
			entries = append(entries, m.isym)
			if m.isym == nil {
				panic("NO ISYM")
			}
			sigs = sigs[1:]
			if len(sigs) == 0 {
				break
			}
		}
	}
	completeItab := len(sigs) == 0
	if !allowNonImplement && !completeItab {
		base.Fatalf("incomplete itab")
	}

	// dump empty itab symbol into i.sym
	// type itab struct {
	//   inter  *interfacetype
	//   _type  *_type
	//   hash   uint32 // copy of _type.hash. Used for type switches.
	//   _      [4]byte
	//   fun    [1]uintptr // variable sized. fun[0]==0 means _type does not implement inter.
	// }
	c := rttype.NewCursor(lsym, 0, rttype.ITab)
	c.Field("Inter").WritePtr(writeType(iface))
	c.Field("Type").WritePtr(writeType(typ))
	c.Field("Hash").WriteUint32(types.TypeHash(typ)) // copy of type hash

	var delta int64
	c = c.Field("Fun")
	if !completeItab {
		// If typ doesn't implement iface, make method entries be zero.
		c.Elem(0).WriteUintptr(0)
	} else {
		var a rttype.ArrayCursor
		a, delta = c.ModifyArray(len(entries))
		for i, fn := range entries {
			a.Elem(i).WritePtrWeak(fn) // method pointer for each method
		}
	}
	// Nothing writes static itabs, so they are read only.
	objw.Global(lsym, int32(rttype.ITab.Size()+delta), int16(obj.DUPOK|obj.RODATA))
	lsym.Set(obj.AttrContentAddressable, true)
}

func WritePluginTable() {
	ptabs := typecheck.Target.PluginExports
	if len(ptabs) == 0 {
		return
	}

	lsym := base.Ctxt.Lookup("go:plugin.tabs")
	ot := 0
	for _, p := range ptabs {
		// Dump ptab symbol into go.pluginsym package.
		//
		// type ptab struct {
		//	name nameOff
		//	typ  typeOff // pointer to symbol
		// }
		nsym := dname(p.Sym().Name, "", nil, true, false)
		t := p.Type()
		if p.Class != ir.PFUNC {
			t = types.NewPtr(t)
		}
		tsym := writeType(t)
		ot = objw.SymPtrOff(lsym, ot, nsym)
		ot = objw.SymPtrOff(lsym, ot, tsym)
		// Plugin exports symbols as interfaces. Mark their types
		// as UsedInIface.
		tsym.Set(obj.AttrUsedInIface, true)
	}
	objw.Global(lsym, int32(ot), int16(obj.RODATA))

	lsym = base.Ctxt.Lookup("go:plugin.exports")
	ot = 0
	for _, p := range ptabs {
		ot = objw.SymPtr(lsym, ot, p.Linksym(), 0)
	}
	objw.Global(lsym, int32(ot), int16(obj.RODATA))
}

// writtenByWriteBasicTypes reports whether typ is written by WriteBasicTypes.
// WriteBasicTypes always writes pointer types; any pointer has been stripped off typ already.
func writtenByWriteBasicTypes(typ *types.Type) bool {
	if typ.Sym() == nil && typ.Kind() == types.TFUNC {
		// func(error) string
		if typ.NumRecvs() == 0 &&
			typ.NumParams() == 1 && typ.NumResults() == 1 &&
			typ.Param(0).Type == types.ErrorType &&
			typ.Result(0).Type == types.Types[types.TSTRING] {
			return true
		}
	}

	// Now we have left the basic types plus any and error, plus slices of them.
	// Strip the slice.
	if typ.Sym() == nil && typ.IsSlice() {
		typ = typ.Elem()
	}

	// Basic types.
	sym := typ.Sym()
	if sym != nil && (sym.Pkg == types.BuiltinPkg || sym.Pkg == types.UnsafePkg) {
		return true
	}
	// any or error
	return (sym == nil && typ.IsEmptyInterface()) || typ == types.ErrorType
}

func WriteBasicTypes() {
	// do basic types if compiling package runtime.
	// they have to be in at least one package,
	// and runtime is always loaded implicitly,
	// so this is as good as any.
	// another possible choice would be package main,
	// but using runtime means fewer copies in object files.
	// The code here needs to be in sync with writtenByWriteBasicTypes above.
	if base.Ctxt.Pkgpath != "runtime" {
		return
	}

	// Note: always write NewPtr(t) because NeedEmit's caller strips the pointer.
	var list []*types.Type
	for i := types.Kind(1); i <= types.TBOOL; i++ {
		list = append(list, types.Types[i])
	}
	list = append(list,
		types.Types[types.TSTRING],
		types.Types[types.TUNSAFEPTR],
		types.AnyType,
		types.ErrorType)
	for _, t := range list {
		writeType(types.NewPtr(t))
		writeType(types.NewPtr(types.NewSlice(t)))
	}

	// emit type for func(error) string,
	// which is the type of an auto-generated wrapper.
	writeType(types.NewPtr(types.NewSignature(nil, []*types.Field{
		types.NewField(base.Pos, nil, types.ErrorType),
	}, []*types.Field{
		types.NewField(base.Pos, nil, types.Types[types.TSTRING]),
	})))
}

type typeAndStr struct {
	t       *types.Type
	short   string // "short" here means TypeSymName
	regular string
}

func typesStrCmp(a, b typeAndStr) int {
	// put named types before unnamed types
	if a.t.Sym() != nil && b.t.Sym() == nil {
		return -1
	}
	if a.t.Sym() == nil && b.t.Sym() != nil {
		return +1
	}

	if r := strings.Compare(a.short, b.short); r != 0 {
		return r
	}
	// When the only difference between the types is whether
	// they refer to byte or uint8, such as **byte vs **uint8,
	// the types' NameStrings can be identical.
	// To preserve deterministic sort ordering, sort these by String().
	//
	// TODO(mdempsky): This all seems suspect. Using LinkString would
	// avoid naming collisions, and there shouldn't be a reason to care
	// about "byte" vs "uint8": they share the same runtime type
	// descriptor anyway.
	if r := strings.Compare(a.regular, b.regular); r != 0 {
		return r
	}
	// Identical anonymous interfaces defined in different locations
	// will be equal for the above checks, but different in DWARF output.
	// Sort by source position to ensure deterministic order.
	// See issues 27013 and 30202.
	if a.t.Kind() == types.TINTER && len(a.t.AllMethods()) > 0 {
		if a.t.AllMethods()[0].Pos.Before(b.t.AllMethods()[0].Pos) {
			return -1
		}
		return +1
	}
	return 0
}

// GCSym returns a data symbol containing GC information for type t.
// GC information is always a bitmask, never a gc program.
// GCSym may be called in concurrent backend, so it does not emit the symbol
// content.
func GCSym(t *types.Type) (lsym *obj.LSym, ptrdata int64) {
	// Record that we need to emit the GC symbol.
	gcsymmu.Lock()
	if _, ok := gcsymset[t]; !ok {
		gcsymset[t] = struct{}{}
	}
	gcsymmu.Unlock()

	lsym, _, ptrdata = dgcsym(t, false, false)
	return
}

// dgcsym returns a data symbol containing GC information for type t, along
// with a boolean reporting whether the gc mask should be computed on demand
// at runtime, and the ptrdata field to record in the reflect type information.
// When write is true, it writes the symbol data.
func dgcsym(t *types.Type, write, onDemandAllowed bool) (lsym *obj.LSym, onDemand bool, ptrdata int64) {
	ptrdata = types.PtrDataSize(t)
	if !onDemandAllowed || ptrdata/int64(types.PtrSize) <= abi.MaxPtrmaskBytes*8 {
		lsym = dgcptrmask(t, write)
		return
	}

	onDemand = true
	lsym = dgcptrmaskOnDemand(t, write)
	return
}

// dgcptrmask emits and returns the symbol containing a pointer mask for type t.
func dgcptrmask(t *types.Type, write bool) *obj.LSym {
	// Bytes we need for the ptrmask.
	n := (types.PtrDataSize(t)/int64(types.PtrSize) + 7) / 8
	// Runtime wants ptrmasks padded to a multiple of uintptr in size.
	n = (n + int64(types.PtrSize) - 1) &^ (int64(types.PtrSize) - 1)
	ptrmask := make([]byte, n)
	fillptrmask(t, ptrmask)
	p := fmt.Sprintf("runtime.gcbits.%x", ptrmask)

	lsym := base.Ctxt.Lookup(p)
	if write && !lsym.OnList() {
		for i, x := range ptrmask {
			objw.Uint8(lsym, i, x)
		}
		objw.Global(lsym, int32(len(ptrmask)), obj.DUPOK|obj.RODATA|obj.LOCAL)
		lsym.Set(obj.AttrContentAddressable, true)
	}
	return lsym
}

// fillptrmask fills in ptrmask with 1s corresponding to the
// word offsets in t that hold pointers.
// ptrmask is assumed to fit at least types.PtrDataSize(t)/PtrSize bits.
func fillptrmask(t *types.Type, ptrmask []byte) {
	for i := range ptrmask {
		ptrmask[i] = 0
	}
	if !t.HasPointers() {
		return
	}

	vec := bitvec.New(8 * int32(len(ptrmask)))
	typebits.Set(t, 0, vec)

	nptr := types.PtrDataSize(t) / int64(types.PtrSize)
	for i := int64(0); i < nptr; i++ {
		if vec.Get(int32(i)) {
			ptrmask[i/8] |= 1 << (uint(i) % 8)
		}
	}
}

// dgcptrmaskOnDemand emits and returns the symbol that should be referenced by
// the GCData field of a type, for large types.
func dgcptrmaskOnDemand(t *types.Type, write bool) *obj.LSym {
	lsym := TypeLinksymPrefix(".gcmask", t)
	if write && !lsym.OnList() {
		// Note: contains a pointer, but a pointer to a
		// persistentalloc allocation. Starts with nil.
		objw.Uintptr(lsym, 0, 0)
		objw.Global(lsym, int32(types.PtrSize), obj.DUPOK|obj.NOPTR|obj.LOCAL) // TODO:bss?
	}
	return lsym
}

// ZeroAddr returns the address of a symbol with at least
// size bytes of zeros.
func ZeroAddr(size int64) ir.Node {
	if size >= 1<<31 {
		base.Fatalf("map elem too big %d", size)
	}
	if ZeroSize < size {
		ZeroSize = size
	}
	lsym := base.PkgLinksym("go:map", "zero", obj.ABI0)
	x := ir.NewLinksymExpr(base.Pos, lsym, types.Types[types.TUINT8])
	return typecheck.Expr(typecheck.NodAddr(x))
}

// NeedEmit reports whether typ is a type that we need to emit code
// for (e.g., runtime type descriptors, method wrappers).
func NeedEmit(typ *types.Type) bool {
	// TODO(mdempsky): Export data should keep track of which anonymous
	// and instantiated types were emitted, so at least downstream
	// packages can skip re-emitting them.
	//
	// Perhaps we can just generalize the linker-symbol indexing to
	// track the index of arbitrary types, not just defined types, and
	// use its presence to detect this. The same idea would work for
	// instantiated generic functions too.

	switch sym := typ.Sym(); {
	case writtenByWriteBasicTypes(typ):
		return base.Ctxt.Pkgpath == "runtime"

	case sym == nil:
		// Anonymous type; possibly never seen before or ever again.
		// Need to emit to be safe (however, see TODO above).
		return true

	case sym.Pkg == types.LocalPkg:
		// Local defined type; our responsibility.
		return true

	case typ.IsFullyInstantiated():
		// Instantiated type; possibly instantiated with unique type arguments.
		// Need to emit to be safe (however, see TODO above).
		return true

	case typ.HasShape():
		// Shape type; need to emit even though it lives in the .shape package.
		// TODO: make sure the linker deduplicates them (see dupok in writeType above).
		return true

	default:
		// Should have been emitted by an imported package.
		return false
	}
}

// Generate a wrapper function to convert from
// a receiver of type T to a receiver of type U.
// That is,
//
//	func (t T) M() {
//		...
//	}
//
// already exists; this function generates
//
//	func (u U) M() {
//		u.M()
//	}
//
// where the types T and U are such that u.M() is valid
// and calls the T.M method.
// The resulting function is for use in method tables.
//
//	rcvr - U
//	method - M func (t T)(), a TFIELD type struct
//
// Also wraps methods on instantiated generic types for use in itab entries.
// For an instantiated generic type G[int], we generate wrappers like:
// G[int] pointer shaped:
//
//	func (x G[int]) f(arg) {
//		.inst.G[int].f(dictionary, x, arg)
//	}
//
// G[int] not pointer shaped:
//
//	func (x *G[int]) f(arg) {
//		.inst.G[int].f(dictionary, *x, arg)
//	}
//
// These wrappers are always fully stenciled.
func methodWrapper(rcvr *types.Type, method *types.Field, forItab bool) *obj.LSym {
	if forItab && !types.IsDirectIface(rcvr) {
		rcvr = rcvr.PtrTo()
	}

	newnam := ir.MethodSym(rcvr, method.Sym)
	lsym := newnam.Linksym()

	// Unified IR creates its own wrappers.
	return lsym
}

var ZeroSize int64

// MarkTypeUsedInInterface marks that type t is converted to an interface.
// This information is used in the linker in dead method elimination.
func MarkTypeUsedInInterface(t *types.Type, from *obj.LSym) {
	if t.HasShape() {
		// Shape types shouldn't be put in interfaces, so we shouldn't ever get here.
		base.Fatalf("shape types have no methods %+v", t)
	}
	MarkTypeSymUsedInInterface(TypeLinksym(t), from)
}
func MarkTypeSymUsedInInterface(tsym *obj.LSym, from *obj.LSym) {
	// Emit a marker relocation. The linker will know the type is converted
	// to an interface if "from" is reachable.
	from.AddRel(base.Ctxt, obj.Reloc{Type: objabi.R_USEIFACE, Sym: tsym})
}

// MarkUsedIfaceMethod marks that an interface method is used in the current
// function. n is OCALLINTER node.
func MarkUsedIfaceMethod(n *ir.CallExpr) {
	// skip unnamed functions (func _())
	if ir.CurFunc.LSym == nil {
		return
	}
	dot := n.Fun.(*ir.SelectorExpr)
	ityp := dot.X.Type()
	if ityp.HasShape() {
		// Here we're calling a method on a generic interface. Something like:
		//
		// type I[T any] interface { foo() T }
		// func f[T any](x I[T]) {
		//     ... = x.foo()
		// }
		// f[int](...)
		// f[string](...)
		//
		// In this case, in f we're calling foo on a generic interface.
		// Which method could that be? Normally we could match the method
		// both by name and by type. But in this case we don't really know
		// the type of the method we're calling. It could be func()int
		// or func()string. So we match on just the function name, instead
		// of both the name and the type used for the non-generic case below.
		// TODO: instantiations at least know the shape of the instantiated
		// type, and the linker could do more complicated matching using
		// some sort of fuzzy shape matching. For now, only use the name
		// of the method for matching.
		ir.CurFunc.LSym.AddRel(base.Ctxt, obj.Reloc{
			Type: objabi.R_USENAMEDMETHOD,
			Sym:  staticdata.StringSymNoCommon(dot.Sel.Name),
		})
		return
	}

	// dot.Offset() is the method index * PtrSize (the offset of code pointer in itab).
	midx := dot.Offset() / int64(types.PtrSize)
	ir.CurFunc.LSym.AddRel(base.Ctxt, obj.Reloc{
		Type: objabi.R_USEIFACEMETHOD,
		Sym:  TypeLinksym(ityp),
		Add:  InterfaceMethodOffset(ityp, midx),
	})
}

func deref(t *types.Type) *types.Type {
	if t.IsPtr() {
		return t.Elem()
	}
	return t
}

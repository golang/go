// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package reflectdata

import (
	"fmt"
	"os"
	"sort"
	"strings"
	"sync"

	"cmd/compile/internal/base"
	"cmd/compile/internal/bitvec"
	"cmd/compile/internal/escape"
	"cmd/compile/internal/inline"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/objw"
	"cmd/compile/internal/typebits"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/internal/gcprog"
	"cmd/internal/obj"
	"cmd/internal/objabi"
	"cmd/internal/src"
)

type itabEntry struct {
	t, itype *types.Type
	lsym     *obj.LSym // symbol of the itab itself

	// symbols of each method in
	// the itab, sorted by byte offset;
	// filled in by CompileITabs
	entries []*obj.LSym
}

type ptabEntry struct {
	s *types.Sym
	t *types.Type
}

func CountTabs() (numPTabs, numITabs int) {
	return len(ptabs), len(itabs)
}

// runtime interface and reflection data structures
var (
	signatmu    sync.Mutex // protects signatset and signatslice
	signatset   = make(map[*types.Type]struct{})
	signatslice []*types.Type

	itabs []itabEntry
	ptabs []*ir.Name
)

type typeSig struct {
	name  *types.Sym
	isym  *obj.LSym
	tsym  *obj.LSym
	type_ *types.Type
	mtype *types.Type
}

// Builds a type representing a Bucket structure for
// the given map type. This type is not visible to users -
// we include only enough information to generate a correct GC
// program for it.
// Make sure this stays in sync with runtime/map.go.
const (
	BUCKETSIZE  = 8
	MAXKEYSIZE  = 128
	MAXELEMSIZE = 128
)

func structfieldSize() int { return 3 * types.PtrSize }       // Sizeof(runtime.structfield{})
func imethodSize() int     { return 4 + 4 }                   // Sizeof(runtime.imethod{})
func commonSize() int      { return 4*types.PtrSize + 8 + 8 } // Sizeof(runtime._type{})

func uncommonSize(t *types.Type) int { // Sizeof(runtime.uncommontype{})
	if t.Sym() == nil && len(methods(t)) == 0 {
		return 0
	}
	return 4 + 2 + 2 + 4 + 4
}

func makefield(name string, t *types.Type) *types.Field {
	sym := (*types.Pkg)(nil).Lookup(name)
	return types.NewField(src.NoXPos, sym, t)
}

// MapBucketType makes the map bucket type given the type of the map.
func MapBucketType(t *types.Type) *types.Type {
	if t.MapType().Bucket != nil {
		return t.MapType().Bucket
	}

	keytype := t.Key()
	elemtype := t.Elem()
	types.CalcSize(keytype)
	types.CalcSize(elemtype)
	if keytype.Width > MAXKEYSIZE {
		keytype = types.NewPtr(keytype)
	}
	if elemtype.Width > MAXELEMSIZE {
		elemtype = types.NewPtr(elemtype)
	}

	field := make([]*types.Field, 0, 5)

	// The first field is: uint8 topbits[BUCKETSIZE].
	arr := types.NewArray(types.Types[types.TUINT8], BUCKETSIZE)
	field = append(field, makefield("topbits", arr))

	arr = types.NewArray(keytype, BUCKETSIZE)
	arr.SetNoalg(true)
	keys := makefield("keys", arr)
	field = append(field, keys)

	arr = types.NewArray(elemtype, BUCKETSIZE)
	arr.SetNoalg(true)
	elems := makefield("elems", arr)
	field = append(field, elems)

	// If keys and elems have no pointers, the map implementation
	// can keep a list of overflow pointers on the side so that
	// buckets can be marked as having no pointers.
	// Arrange for the bucket to have no pointers by changing
	// the type of the overflow field to uintptr in this case.
	// See comment on hmap.overflow in runtime/map.go.
	otyp := types.Types[types.TUNSAFEPTR]
	if !elemtype.HasPointers() && !keytype.HasPointers() {
		otyp = types.Types[types.TUINTPTR]
	}
	overflow := makefield("overflow", otyp)
	field = append(field, overflow)

	// link up fields
	bucket := types.NewStruct(types.NoPkg, field[:])
	bucket.SetNoalg(true)
	types.CalcSize(bucket)

	// Check invariants that map code depends on.
	if !types.IsComparable(t.Key()) {
		base.Fatalf("unsupported map key type for %v", t)
	}
	if BUCKETSIZE < 8 {
		base.Fatalf("bucket size too small for proper alignment")
	}
	if keytype.Align > BUCKETSIZE {
		base.Fatalf("key align too big for %v", t)
	}
	if elemtype.Align > BUCKETSIZE {
		base.Fatalf("elem align too big for %v", t)
	}
	if keytype.Width > MAXKEYSIZE {
		base.Fatalf("key size to large for %v", t)
	}
	if elemtype.Width > MAXELEMSIZE {
		base.Fatalf("elem size to large for %v", t)
	}
	if t.Key().Width > MAXKEYSIZE && !keytype.IsPtr() {
		base.Fatalf("key indirect incorrect for %v", t)
	}
	if t.Elem().Width > MAXELEMSIZE && !elemtype.IsPtr() {
		base.Fatalf("elem indirect incorrect for %v", t)
	}
	if keytype.Width%int64(keytype.Align) != 0 {
		base.Fatalf("key size not a multiple of key align for %v", t)
	}
	if elemtype.Width%int64(elemtype.Align) != 0 {
		base.Fatalf("elem size not a multiple of elem align for %v", t)
	}
	if bucket.Align%keytype.Align != 0 {
		base.Fatalf("bucket align not multiple of key align %v", t)
	}
	if bucket.Align%elemtype.Align != 0 {
		base.Fatalf("bucket align not multiple of elem align %v", t)
	}
	if keys.Offset%int64(keytype.Align) != 0 {
		base.Fatalf("bad alignment of keys in bmap for %v", t)
	}
	if elems.Offset%int64(elemtype.Align) != 0 {
		base.Fatalf("bad alignment of elems in bmap for %v", t)
	}

	// Double-check that overflow field is final memory in struct,
	// with no padding at end.
	if overflow.Offset != bucket.Width-int64(types.PtrSize) {
		base.Fatalf("bad offset of overflow in bmap for %v", t)
	}

	t.MapType().Bucket = bucket

	bucket.StructType().Map = t
	return bucket
}

// MapType builds a type representing a Hmap structure for the given map type.
// Make sure this stays in sync with runtime/map.go.
func MapType(t *types.Type) *types.Type {
	if t.MapType().Hmap != nil {
		return t.MapType().Hmap
	}

	bmap := MapBucketType(t)

	// build a struct:
	// type hmap struct {
	//    count      int
	//    flags      uint8
	//    B          uint8
	//    noverflow  uint16
	//    hash0      uint32
	//    buckets    *bmap
	//    oldbuckets *bmap
	//    nevacuate  uintptr
	//    extra      unsafe.Pointer // *mapextra
	// }
	// must match runtime/map.go:hmap.
	fields := []*types.Field{
		makefield("count", types.Types[types.TINT]),
		makefield("flags", types.Types[types.TUINT8]),
		makefield("B", types.Types[types.TUINT8]),
		makefield("noverflow", types.Types[types.TUINT16]),
		makefield("hash0", types.Types[types.TUINT32]), // Used in walk.go for OMAKEMAP.
		makefield("buckets", types.NewPtr(bmap)),       // Used in walk.go for OMAKEMAP.
		makefield("oldbuckets", types.NewPtr(bmap)),
		makefield("nevacuate", types.Types[types.TUINTPTR]),
		makefield("extra", types.Types[types.TUNSAFEPTR]),
	}

	hmap := types.NewStruct(types.NoPkg, fields)
	hmap.SetNoalg(true)
	types.CalcSize(hmap)

	// The size of hmap should be 48 bytes on 64 bit
	// and 28 bytes on 32 bit platforms.
	if size := int64(8 + 5*types.PtrSize); hmap.Width != size {
		base.Fatalf("hmap size not correct: got %d, want %d", hmap.Width, size)
	}

	t.MapType().Hmap = hmap
	hmap.StructType().Map = t
	return hmap
}

// MapIterType builds a type representing an Hiter structure for the given map type.
// Make sure this stays in sync with runtime/map.go.
func MapIterType(t *types.Type) *types.Type {
	if t.MapType().Hiter != nil {
		return t.MapType().Hiter
	}

	hmap := MapType(t)
	bmap := MapBucketType(t)

	// build a struct:
	// type hiter struct {
	//    key         *Key
	//    elem        *Elem
	//    t           unsafe.Pointer // *MapType
	//    h           *hmap
	//    buckets     *bmap
	//    bptr        *bmap
	//    overflow    unsafe.Pointer // *[]*bmap
	//    oldoverflow unsafe.Pointer // *[]*bmap
	//    startBucket uintptr
	//    offset      uint8
	//    wrapped     bool
	//    B           uint8
	//    i           uint8
	//    bucket      uintptr
	//    checkBucket uintptr
	// }
	// must match runtime/map.go:hiter.
	fields := []*types.Field{
		makefield("key", types.NewPtr(t.Key())),   // Used in range.go for TMAP.
		makefield("elem", types.NewPtr(t.Elem())), // Used in range.go for TMAP.
		makefield("t", types.Types[types.TUNSAFEPTR]),
		makefield("h", types.NewPtr(hmap)),
		makefield("buckets", types.NewPtr(bmap)),
		makefield("bptr", types.NewPtr(bmap)),
		makefield("overflow", types.Types[types.TUNSAFEPTR]),
		makefield("oldoverflow", types.Types[types.TUNSAFEPTR]),
		makefield("startBucket", types.Types[types.TUINTPTR]),
		makefield("offset", types.Types[types.TUINT8]),
		makefield("wrapped", types.Types[types.TBOOL]),
		makefield("B", types.Types[types.TUINT8]),
		makefield("i", types.Types[types.TUINT8]),
		makefield("bucket", types.Types[types.TUINTPTR]),
		makefield("checkBucket", types.Types[types.TUINTPTR]),
	}

	// build iterator struct holding the above fields
	hiter := types.NewStruct(types.NoPkg, fields)
	hiter.SetNoalg(true)
	types.CalcSize(hiter)
	if hiter.Width != int64(12*types.PtrSize) {
		base.Fatalf("hash_iter size not correct %d %d", hiter.Width, 12*types.PtrSize)
	}
	t.MapType().Hiter = hiter
	hiter.StructType().Map = t
	return hiter
}

// methods returns the methods of the non-interface type t, sorted by name.
// Generates stub functions as needed.
func methods(t *types.Type) []*typeSig {
	// method type
	mt := types.ReceiverBaseType(t)

	if mt == nil {
		return nil
	}
	typecheck.CalcMethods(mt)

	// type stored in interface word
	it := t

	if !types.IsDirectIface(it) {
		it = types.NewPtr(t)
	}

	// make list of methods for t,
	// generating code if necessary.
	var ms []*typeSig
	for _, f := range mt.AllMethods().Slice() {
		if f.Sym == nil {
			base.Fatalf("method with no sym on %v", mt)
		}
		if !f.IsMethod() {
			base.Fatalf("non-method on %v method %v %v", mt, f.Sym, f)
		}
		if f.Type.Recv() == nil {
			base.Fatalf("receiver with no type on %v method %v %v", mt, f.Sym, f)
		}
		if f.Nointerface() {
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
			isym:  methodWrapper(it, f),
			tsym:  methodWrapper(t, f),
			type_: typecheck.NewMethodType(f.Type, t),
			mtype: typecheck.NewMethodType(f.Type, nil),
		}
		ms = append(ms, sig)
	}

	return ms
}

// imethods returns the methods of the interface type t, sorted by name.
func imethods(t *types.Type) []*typeSig {
	var methods []*typeSig
	for _, f := range t.AllMethods().Slice() {
		if f.Type.Kind() != types.TFUNC || f.Sym == nil {
			continue
		}
		if f.Sym.IsBlank() {
			base.Fatalf("unexpected blank symbol in interface method set")
		}
		if n := len(methods); n > 0 {
			last := methods[n-1]
			if !last.name.Less(f.Sym) {
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
		methodWrapper(t, f)
	}

	return methods
}

func dimportpath(p *types.Pkg) {
	if p.Pathsym != nil {
		return
	}

	// If we are compiling the runtime package, there are two runtime packages around
	// -- localpkg and Pkgs.Runtime. We don't want to produce import path symbols for
	// both of them, so just produce one for localpkg.
	if base.Ctxt.Pkgpath == "runtime" && p == ir.Pkgs.Runtime {
		return
	}

	str := p.Path
	if p == types.LocalPkg {
		// Note: myimportpath != "", or else dgopkgpath won't call dimportpath.
		str = base.Ctxt.Pkgpath
	}

	s := base.Ctxt.Lookup("type..importpath." + p.Prefix + ".")
	ot := dnameData(s, 0, str, "", nil, false)
	objw.Global(s, int32(ot), obj.DUPOK|obj.RODATA)
	s.Set(obj.AttrContentAddressable, true)
	p.Pathsym = s
}

func dgopkgpath(s *obj.LSym, ot int, pkg *types.Pkg) int {
	if pkg == nil {
		return objw.Uintptr(s, ot, 0)
	}

	if pkg == types.LocalPkg && base.Ctxt.Pkgpath == "" {
		// If we don't know the full import path of the package being compiled
		// (i.e. -p was not passed on the compiler command line), emit a reference to
		// type..importpath.""., which the linker will rewrite using the correct import path.
		// Every package that imports this one directly defines the symbol.
		// See also https://groups.google.com/forum/#!topic/golang-dev/myb9s53HxGQ.
		ns := base.Ctxt.Lookup(`type..importpath."".`)
		return objw.SymPtr(s, ot, ns, 0)
	}

	dimportpath(pkg)
	return objw.SymPtr(s, ot, pkg.Pathsym, 0)
}

// dgopkgpathOff writes an offset relocation in s at offset ot to the pkg path symbol.
func dgopkgpathOff(s *obj.LSym, ot int, pkg *types.Pkg) int {
	if pkg == nil {
		return objw.Uint32(s, ot, 0)
	}
	if pkg == types.LocalPkg && base.Ctxt.Pkgpath == "" {
		// If we don't know the full import path of the package being compiled
		// (i.e. -p was not passed on the compiler command line), emit a reference to
		// type..importpath.""., which the linker will rewrite using the correct import path.
		// Every package that imports this one directly defines the symbol.
		// See also https://groups.google.com/forum/#!topic/golang-dev/myb9s53HxGQ.
		ns := base.Ctxt.Lookup(`type..importpath."".`)
		return objw.SymPtrOff(s, ot, ns)
	}

	dimportpath(pkg)
	return objw.SymPtrOff(s, ot, pkg.Pathsym)
}

// dnameField dumps a reflect.name for a struct field.
func dnameField(lsym *obj.LSym, ot int, spkg *types.Pkg, ft *types.Field) int {
	if !types.IsExported(ft.Sym.Name) && ft.Sym.Pkg != spkg {
		base.Fatalf("package mismatch for %v", ft.Sym)
	}
	nsym := dname(ft.Sym.Name, ft.Note, nil, types.IsExported(ft.Sym.Name))
	return objw.SymPtr(lsym, ot, nsym, 0)
}

// dnameData writes the contents of a reflect.name into s at offset ot.
func dnameData(s *obj.LSym, ot int, name, tag string, pkg *types.Pkg, exported bool) int {
	if len(name) > 1<<16-1 {
		base.Fatalf("name too long: %s", name)
	}
	if len(tag) > 1<<16-1 {
		base.Fatalf("tag too long: %s", tag)
	}

	// Encode name and tag. See reflect/type.go for details.
	var bits byte
	l := 1 + 2 + len(name)
	if exported {
		bits |= 1 << 0
	}
	if len(tag) > 0 {
		l += 2 + len(tag)
		bits |= 1 << 1
	}
	if pkg != nil {
		bits |= 1 << 2
	}
	b := make([]byte, l)
	b[0] = bits
	b[1] = uint8(len(name) >> 8)
	b[2] = uint8(len(name))
	copy(b[3:], name)
	if len(tag) > 0 {
		tb := b[3+len(name):]
		tb[0] = uint8(len(tag) >> 8)
		tb[1] = uint8(len(tag))
		copy(tb[2:], tag)
	}

	ot = int(s.WriteBytes(base.Ctxt, int64(ot), b))

	if pkg != nil {
		ot = dgopkgpathOff(s, ot, pkg)
	}

	return ot
}

var dnameCount int

// dname creates a reflect.name for a struct field or method.
func dname(name, tag string, pkg *types.Pkg, exported bool) *obj.LSym {
	// Write out data as "type.." to signal two things to the
	// linker, first that when dynamically linking, the symbol
	// should be moved to a relro section, and second that the
	// contents should not be decoded as a type.
	sname := "type..namedata."
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
		sname = fmt.Sprintf(`%s"".%d`, sname, dnameCount)
		dnameCount++
	}
	s := base.Ctxt.Lookup(sname)
	if len(s.P) > 0 {
		return s
	}
	ot := dnameData(s, 0, name, tag, pkg, exported)
	objw.Global(s, int32(ot), obj.DUPOK|obj.RODATA)
	s.Set(obj.AttrContentAddressable, true)
	return s
}

// dextratype dumps the fields of a runtime.uncommontype.
// dataAdd is the offset in bytes after the header where the
// backing array of the []method field is written (by dextratypeData).
func dextratype(lsym *obj.LSym, ot int, t *types.Type, dataAdd int) int {
	m := methods(t)
	if t.Sym() == nil && len(m) == 0 {
		return ot
	}
	noff := int(types.Rnd(int64(ot), int64(types.PtrSize)))
	if noff != ot {
		base.Fatalf("unexpected alignment in dextratype for %v", t)
	}

	for _, a := range m {
		writeType(a.type_)
	}

	ot = dgopkgpathOff(lsym, ot, typePkg(t))

	dataAdd += uncommonSize(t)
	mcount := len(m)
	if mcount != int(uint16(mcount)) {
		base.Fatalf("too many methods on %v: %d", t, mcount)
	}
	xcount := sort.Search(mcount, func(i int) bool { return !types.IsExported(m[i].name.Name) })
	if dataAdd != int(uint32(dataAdd)) {
		base.Fatalf("methods are too far away on %v: %d", t, dataAdd)
	}

	ot = objw.Uint16(lsym, ot, uint16(mcount))
	ot = objw.Uint16(lsym, ot, uint16(xcount))
	ot = objw.Uint32(lsym, ot, uint32(dataAdd))
	ot = objw.Uint32(lsym, ot, 0)
	return ot
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
	if tsym != nil && t != types.Types[t.Kind()] && t != types.ErrorType {
		return tsym.Pkg
	}
	return nil
}

// dextratypeData dumps the backing array for the []method field of
// runtime.uncommontype.
func dextratypeData(lsym *obj.LSym, ot int, t *types.Type) int {
	for _, a := range methods(t) {
		// ../../../../runtime/type.go:/method
		exported := types.IsExported(a.name.Name)
		var pkg *types.Pkg
		if !exported && a.name.Pkg != typePkg(t) {
			pkg = a.name.Pkg
		}
		nsym := dname(a.name.Name, "", pkg, exported)

		ot = objw.SymPtrOff(lsym, ot, nsym)
		ot = dmethodptrOff(lsym, ot, writeType(a.mtype))
		ot = dmethodptrOff(lsym, ot, a.isym)
		ot = dmethodptrOff(lsym, ot, a.tsym)
	}
	return ot
}

func dmethodptrOff(s *obj.LSym, ot int, x *obj.LSym) int {
	objw.Uint32(s, ot, 0)
	r := obj.Addrel(s)
	r.Off = int32(ot)
	r.Siz = 4
	r.Sym = x
	r.Type = objabi.R_METHODOFF
	return ot + 4
}

var kinds = []int{
	types.TINT:        objabi.KindInt,
	types.TUINT:       objabi.KindUint,
	types.TINT8:       objabi.KindInt8,
	types.TUINT8:      objabi.KindUint8,
	types.TINT16:      objabi.KindInt16,
	types.TUINT16:     objabi.KindUint16,
	types.TINT32:      objabi.KindInt32,
	types.TUINT32:     objabi.KindUint32,
	types.TINT64:      objabi.KindInt64,
	types.TUINT64:     objabi.KindUint64,
	types.TUINTPTR:    objabi.KindUintptr,
	types.TFLOAT32:    objabi.KindFloat32,
	types.TFLOAT64:    objabi.KindFloat64,
	types.TBOOL:       objabi.KindBool,
	types.TSTRING:     objabi.KindString,
	types.TPTR:        objabi.KindPtr,
	types.TSTRUCT:     objabi.KindStruct,
	types.TINTER:      objabi.KindInterface,
	types.TCHAN:       objabi.KindChan,
	types.TMAP:        objabi.KindMap,
	types.TARRAY:      objabi.KindArray,
	types.TSLICE:      objabi.KindSlice,
	types.TFUNC:       objabi.KindFunc,
	types.TCOMPLEX64:  objabi.KindComplex64,
	types.TCOMPLEX128: objabi.KindComplex128,
	types.TUNSAFEPTR:  objabi.KindUnsafePointer,
}

// tflag is documented in reflect/type.go.
//
// tflag values must be kept in sync with copies in:
//	cmd/compile/internal/gc/reflect.go
//	cmd/link/internal/ld/decodesym.go
//	reflect/type.go
//	runtime/type.go
const (
	tflagUncommon      = 1 << 0
	tflagExtraStar     = 1 << 1
	tflagNamed         = 1 << 2
	tflagRegularMemory = 1 << 3
)

var (
	memhashvarlen  *obj.LSym
	memequalvarlen *obj.LSym
)

// dcommontype dumps the contents of a reflect.rtype (runtime._type).
func dcommontype(lsym *obj.LSym, t *types.Type) int {
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

	gcsym, useGCProg, ptrdata := dgcsym(t)

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
	ot := 0
	ot = objw.Uintptr(lsym, ot, uint64(t.Width))
	ot = objw.Uintptr(lsym, ot, uint64(ptrdata))
	ot = objw.Uint32(lsym, ot, types.TypeHash(t))

	var tflag uint8
	if uncommonSize(t) != 0 {
		tflag |= tflagUncommon
	}
	if t.Sym() != nil && t.Sym().Name != "" {
		tflag |= tflagNamed
	}
	if isRegularMemory(t) {
		tflag |= tflagRegularMemory
	}

	exported := false
	p := t.LongString()
	// If we're writing out type T,
	// we are very likely to write out type *T as well.
	// Use the string "*T"[1:] for "T", so that the two
	// share storage. This is a cheap way to reduce the
	// amount of space taken up by reflect strings.
	if !strings.HasPrefix(p, "*") {
		p = "*" + p
		tflag |= tflagExtraStar
		if t.Sym() != nil {
			exported = types.IsExported(t.Sym().Name)
		}
	} else {
		if t.Elem() != nil && t.Elem().Sym() != nil {
			exported = types.IsExported(t.Elem().Sym().Name)
		}
	}

	ot = objw.Uint8(lsym, ot, tflag)

	// runtime (and common sense) expects alignment to be a power of two.
	i := int(t.Align)

	if i == 0 {
		i = 1
	}
	if i&(i-1) != 0 {
		base.Fatalf("invalid alignment %d for %v", t.Align, t)
	}
	ot = objw.Uint8(lsym, ot, t.Align) // align
	ot = objw.Uint8(lsym, ot, t.Align) // fieldAlign

	i = kinds[t.Kind()]
	if types.IsDirectIface(t) {
		i |= objabi.KindDirectIface
	}
	if useGCProg {
		i |= objabi.KindGCProg
	}
	ot = objw.Uint8(lsym, ot, uint8(i)) // kind
	if eqfunc != nil {
		ot = objw.SymPtr(lsym, ot, eqfunc, 0) // equality function
	} else {
		ot = objw.Uintptr(lsym, ot, 0) // type we can't do == with
	}
	ot = objw.SymPtr(lsym, ot, gcsym, 0) // gcdata

	nsym := dname(p, "", nil, exported)
	ot = objw.SymPtrOff(lsym, ot, nsym) // str
	// ptrToThis
	if sptr == nil {
		ot = objw.Uint32(lsym, ot, 0)
	} else if sptrWeak {
		ot = objw.SymPtrWeakOff(lsym, ot, sptr)
	} else {
		ot = objw.SymPtrOff(lsym, ot, sptr)
	}

	return ot
}

// TrackSym returns the symbol for tracking use of field/method f, assumed
// to be a member of struct/interface type t.
func TrackSym(t *types.Type, f *types.Field) *obj.LSym {
	return base.PkgLinksym("go.track", t.ShortString()+"."+f.Sym.Name, obj.ABI0)
}

func TypeSymPrefix(prefix string, t *types.Type) *types.Sym {
	p := prefix + "." + t.ShortString()
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
	return TypeSym(t).Linksym()
}

func TypePtr(t *types.Type) *ir.AddrExpr {
	n := ir.NewLinksymExpr(base.Pos, TypeLinksym(t), types.Types[types.TUINT8])
	return typecheck.Expr(typecheck.NodAddr(n)).(*ir.AddrExpr)
}

func ITabAddr(t, itype *types.Type) *ir.AddrExpr {
	if t == nil || (t.IsPtr() && t.Elem() == nil) || t.IsUntyped() || !itype.IsInterface() || itype.IsEmptyInterface() {
		base.Fatalf("ITabAddr(%v, %v)", t, itype)
	}
	s, existed := ir.Pkgs.Itab.LookupOK(t.ShortString() + "," + itype.ShortString())
	if !existed {
		itabs = append(itabs, itabEntry{t: t, itype: itype, lsym: s.Linksym()})
	}

	lsym := s.Linksym()
	n := ir.NewLinksymExpr(base.Pos, lsym, types.Types[types.TUINT8])
	return typecheck.Expr(typecheck.NodAddr(n)).(*ir.AddrExpr)
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
		for _, t1 := range t.Fields().Slice() {
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
		for _, t1 := range t.Fields().Slice() {
			if hashMightPanic(t1.Type) {
				return true
			}
		}
		return false

	default:
		return false
	}
}

// formalType replaces byte and rune aliases with real types.
// They've been separate internally to make error messages
// better, but we have to merge them in the reflect tables.
func formalType(t *types.Type) *types.Type {
	if t == types.ByteType || t == types.RuneType {
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
	if s.Siggen() {
		return lsym
	}
	s.SetSiggen(true)

	// special case (look for runtime below):
	// when compiling package runtime,
	// emit the type structures for int, float, etc.
	tbase := t

	if t.IsPtr() && t.Sym() == nil && t.Elem().Sym() != nil {
		tbase = t.Elem()
	}
	dupok := 0
	if tbase.Sym() == nil {
		dupok = obj.DUPOK
	}

	if base.Ctxt.Pkgpath != "runtime" || (tbase != types.Types[tbase.Kind()] && tbase != types.ByteType && tbase != types.RuneType && tbase != types.ErrorType) { // int, float, etc
		// named types from other files are defined only by those files
		if tbase.Sym() != nil && tbase.Sym().Pkg != types.LocalPkg {
			if i := typecheck.BaseTypeIndex(t); i >= 0 {
				lsym.Pkg = tbase.Sym().Pkg.Prefix
				lsym.SymIdx = int32(i)
				lsym.Set(obj.AttrIndexed, true)
			}
			return lsym
		}
		// TODO(mdempsky): Investigate whether this can happen.
		if tbase.Kind() == types.TFORW {
			return lsym
		}
	}

	ot := 0
	switch t.Kind() {
	default:
		ot = dcommontype(lsym, t)
		ot = dextratype(lsym, ot, t, 0)

	case types.TARRAY:
		// ../../../../runtime/type.go:/arrayType
		s1 := writeType(t.Elem())
		t2 := types.NewSlice(t.Elem())
		s2 := writeType(t2)
		ot = dcommontype(lsym, t)
		ot = objw.SymPtr(lsym, ot, s1, 0)
		ot = objw.SymPtr(lsym, ot, s2, 0)
		ot = objw.Uintptr(lsym, ot, uint64(t.NumElem()))
		ot = dextratype(lsym, ot, t, 0)

	case types.TSLICE:
		// ../../../../runtime/type.go:/sliceType
		s1 := writeType(t.Elem())
		ot = dcommontype(lsym, t)
		ot = objw.SymPtr(lsym, ot, s1, 0)
		ot = dextratype(lsym, ot, t, 0)

	case types.TCHAN:
		// ../../../../runtime/type.go:/chanType
		s1 := writeType(t.Elem())
		ot = dcommontype(lsym, t)
		ot = objw.SymPtr(lsym, ot, s1, 0)
		ot = objw.Uintptr(lsym, ot, uint64(t.ChanDir()))
		ot = dextratype(lsym, ot, t, 0)

	case types.TFUNC:
		for _, t1 := range t.Recvs().Fields().Slice() {
			writeType(t1.Type)
		}
		isddd := false
		for _, t1 := range t.Params().Fields().Slice() {
			isddd = t1.IsDDD()
			writeType(t1.Type)
		}
		for _, t1 := range t.Results().Fields().Slice() {
			writeType(t1.Type)
		}

		ot = dcommontype(lsym, t)
		inCount := t.NumRecvs() + t.NumParams()
		outCount := t.NumResults()
		if isddd {
			outCount |= 1 << 15
		}
		ot = objw.Uint16(lsym, ot, uint16(inCount))
		ot = objw.Uint16(lsym, ot, uint16(outCount))
		if types.PtrSize == 8 {
			ot += 4 // align for *rtype
		}

		dataAdd := (inCount + t.NumResults()) * types.PtrSize
		ot = dextratype(lsym, ot, t, dataAdd)

		// Array of rtype pointers follows funcType.
		for _, t1 := range t.Recvs().Fields().Slice() {
			ot = objw.SymPtr(lsym, ot, writeType(t1.Type), 0)
		}
		for _, t1 := range t.Params().Fields().Slice() {
			ot = objw.SymPtr(lsym, ot, writeType(t1.Type), 0)
		}
		for _, t1 := range t.Results().Fields().Slice() {
			ot = objw.SymPtr(lsym, ot, writeType(t1.Type), 0)
		}

	case types.TINTER:
		m := imethods(t)
		n := len(m)
		for _, a := range m {
			writeType(a.type_)
		}

		// ../../../../runtime/type.go:/interfaceType
		ot = dcommontype(lsym, t)

		var tpkg *types.Pkg
		if t.Sym() != nil && t != types.Types[t.Kind()] && t != types.ErrorType {
			tpkg = t.Sym().Pkg
		}
		ot = dgopkgpath(lsym, ot, tpkg)

		ot = objw.SymPtr(lsym, ot, lsym, ot+3*types.PtrSize+uncommonSize(t))
		ot = objw.Uintptr(lsym, ot, uint64(n))
		ot = objw.Uintptr(lsym, ot, uint64(n))
		dataAdd := imethodSize() * n
		ot = dextratype(lsym, ot, t, dataAdd)

		for _, a := range m {
			// ../../../../runtime/type.go:/imethod
			exported := types.IsExported(a.name.Name)
			var pkg *types.Pkg
			if !exported && a.name.Pkg != tpkg {
				pkg = a.name.Pkg
			}
			nsym := dname(a.name.Name, "", pkg, exported)

			ot = objw.SymPtrOff(lsym, ot, nsym)
			ot = objw.SymPtrOff(lsym, ot, writeType(a.type_))
		}

	// ../../../../runtime/type.go:/mapType
	case types.TMAP:
		s1 := writeType(t.Key())
		s2 := writeType(t.Elem())
		s3 := writeType(MapBucketType(t))
		hasher := genhash(t.Key())

		ot = dcommontype(lsym, t)
		ot = objw.SymPtr(lsym, ot, s1, 0)
		ot = objw.SymPtr(lsym, ot, s2, 0)
		ot = objw.SymPtr(lsym, ot, s3, 0)
		ot = objw.SymPtr(lsym, ot, hasher, 0)
		var flags uint32
		// Note: flags must match maptype accessors in ../../../../runtime/type.go
		// and maptype builder in ../../../../reflect/type.go:MapOf.
		if t.Key().Width > MAXKEYSIZE {
			ot = objw.Uint8(lsym, ot, uint8(types.PtrSize))
			flags |= 1 // indirect key
		} else {
			ot = objw.Uint8(lsym, ot, uint8(t.Key().Width))
		}

		if t.Elem().Width > MAXELEMSIZE {
			ot = objw.Uint8(lsym, ot, uint8(types.PtrSize))
			flags |= 2 // indirect value
		} else {
			ot = objw.Uint8(lsym, ot, uint8(t.Elem().Width))
		}
		ot = objw.Uint16(lsym, ot, uint16(MapBucketType(t).Width))
		if types.IsReflexive(t.Key()) {
			flags |= 4 // reflexive key
		}
		if needkeyupdate(t.Key()) {
			flags |= 8 // need key update
		}
		if hashMightPanic(t.Key()) {
			flags |= 16 // hash might panic
		}
		ot = objw.Uint32(lsym, ot, flags)
		ot = dextratype(lsym, ot, t, 0)

	case types.TPTR:
		if t.Elem().Kind() == types.TANY {
			// ../../../../runtime/type.go:/UnsafePointerType
			ot = dcommontype(lsym, t)
			ot = dextratype(lsym, ot, t, 0)

			break
		}

		// ../../../../runtime/type.go:/ptrType
		s1 := writeType(t.Elem())

		ot = dcommontype(lsym, t)
		ot = objw.SymPtr(lsym, ot, s1, 0)
		ot = dextratype(lsym, ot, t, 0)

	// ../../../../runtime/type.go:/structType
	// for security, only the exported fields.
	case types.TSTRUCT:
		fields := t.Fields().Slice()
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

		ot = dcommontype(lsym, t)
		ot = dgopkgpath(lsym, ot, spkg)
		ot = objw.SymPtr(lsym, ot, lsym, ot+3*types.PtrSize+uncommonSize(t))
		ot = objw.Uintptr(lsym, ot, uint64(len(fields)))
		ot = objw.Uintptr(lsym, ot, uint64(len(fields)))

		dataAdd := len(fields) * structfieldSize()
		ot = dextratype(lsym, ot, t, dataAdd)

		for _, f := range fields {
			// ../../../../runtime/type.go:/structField
			ot = dnameField(lsym, ot, spkg, f)
			ot = objw.SymPtr(lsym, ot, writeType(f.Type), 0)
			offsetAnon := uint64(f.Offset) << 1
			if offsetAnon>>1 != uint64(f.Offset) {
				base.Fatalf("%v: bad field offset for %s", t, f.Sym.Name)
			}
			if f.Embedded != 0 {
				offsetAnon |= 1
			}
			ot = objw.Uintptr(lsym, ot, offsetAnon)
		}
	}

	ot = dextratypeData(lsym, ot, t)
	objw.Global(lsym, int32(ot), int16(dupok|obj.RODATA))

	// The linker will leave a table of all the typelinks for
	// types in the binary, so the runtime can find them.
	//
	// When buildmode=shared, all types are in typelinks so the
	// runtime can deduplicate type pointers.
	keep := base.Ctxt.Flag_dynlink
	if !keep && t.Sym() == nil {
		// For an unnamed type, we only need the link if the type can
		// be created at run time by reflect.PtrTo and similar
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

// for each itabEntry, gather the methods on
// the concrete type that implement the interface
func CompileITabs() {
	for i := range itabs {
		tab := &itabs[i]
		methods := genfun(tab.t, tab.itype)
		if len(methods) == 0 {
			continue
		}
		tab.entries = methods
	}
}

// for the given concrete type and interface
// type, return the (sorted) set of methods
// on the concrete type that implement the interface
func genfun(t, it *types.Type) []*obj.LSym {
	if t == nil || it == nil {
		return nil
	}
	sigs := imethods(it)
	methods := methods(t)
	out := make([]*obj.LSym, 0, len(sigs))
	// TODO(mdempsky): Short circuit before calling methods(t)?
	// See discussion on CL 105039.
	if len(sigs) == 0 {
		return nil
	}

	// both sigs and methods are sorted by name,
	// so we can find the intersect in a single pass
	for _, m := range methods {
		if m.name == sigs[0].name {
			out = append(out, m.isym)
			sigs = sigs[1:]
			if len(sigs) == 0 {
				break
			}
		}
	}

	if len(sigs) != 0 {
		base.Fatalf("incomplete itab")
	}

	return out
}

// ITabSym uses the information gathered in
// CompileITabs to de-virtualize interface methods.
// Since this is called by the SSA backend, it shouldn't
// generate additional Nodes, Syms, etc.
func ITabSym(it *obj.LSym, offset int64) *obj.LSym {
	var syms []*obj.LSym
	if it == nil {
		return nil
	}

	for i := range itabs {
		e := &itabs[i]
		if e.lsym == it {
			syms = e.entries
			break
		}
	}
	if syms == nil {
		return nil
	}

	// keep this arithmetic in sync with *itab layout
	methodnum := int((offset - 2*int64(types.PtrSize) - 8) / int64(types.PtrSize))
	if methodnum >= len(syms) {
		return nil
	}
	return syms[methodnum]
}

// NeedRuntimeType ensures that a runtime type descriptor is emitted for t.
func NeedRuntimeType(t *types.Type) {
	if t.HasTParam() {
		// Generic types don't have a runtime type descriptor (but will
		// have a dictionary)
		return
	}
	if _, ok := signatset[t]; !ok {
		signatset[t] = struct{}{}
		signatslice = append(signatslice, t)
	}
}

func WriteRuntimeTypes() {
	// Process signatset. Use a loop, as writeType adds
	// entries to signatset while it is being processed.
	signats := make([]typeAndStr, len(signatslice))
	for len(signatslice) > 0 {
		signats = signats[:0]
		// Transfer entries to a slice and sort, for reproducible builds.
		for _, t := range signatslice {
			signats = append(signats, typeAndStr{t: t, short: types.TypeSymName(t), regular: t.String()})
			delete(signatset, t)
		}
		signatslice = signatslice[:0]
		sort.Sort(typesByString(signats))
		for _, ts := range signats {
			t := ts.t
			writeType(t)
			if t.Sym() != nil {
				writeType(types.NewPtr(t))
			}
		}
	}
}

func WriteTabs() {
	// process itabs
	for _, i := range itabs {
		// dump empty itab symbol into i.sym
		// type itab struct {
		//   inter  *interfacetype
		//   _type  *_type
		//   hash   uint32
		//   _      [4]byte
		//   fun    [1]uintptr // variable sized
		// }
		o := objw.SymPtr(i.lsym, 0, writeType(i.itype), 0)
		o = objw.SymPtr(i.lsym, o, writeType(i.t), 0)
		o = objw.Uint32(i.lsym, o, types.TypeHash(i.t)) // copy of type hash
		o += 4                                          // skip unused field
		for _, fn := range genfun(i.t, i.itype) {
			o = objw.SymPtrWeak(i.lsym, o, fn, 0) // method pointer for each method
		}
		// Nothing writes static itabs, so they are read only.
		objw.Global(i.lsym, int32(o), int16(obj.DUPOK|obj.RODATA))
		i.lsym.Set(obj.AttrContentAddressable, true)
	}

	// process ptabs
	if types.LocalPkg.Name == "main" && len(ptabs) > 0 {
		ot := 0
		s := base.Ctxt.Lookup("go.plugin.tabs")
		for _, p := range ptabs {
			// Dump ptab symbol into go.pluginsym package.
			//
			// type ptab struct {
			//	name nameOff
			//	typ  typeOff // pointer to symbol
			// }
			nsym := dname(p.Sym().Name, "", nil, true)
			t := p.Type()
			if p.Class != ir.PFUNC {
				t = types.NewPtr(t)
			}
			tsym := writeType(t)
			ot = objw.SymPtrOff(s, ot, nsym)
			ot = objw.SymPtrOff(s, ot, tsym)
			// Plugin exports symbols as interfaces. Mark their types
			// as UsedInIface.
			tsym.Set(obj.AttrUsedInIface, true)
		}
		objw.Global(s, int32(ot), int16(obj.RODATA))

		ot = 0
		s = base.Ctxt.Lookup("go.plugin.exports")
		for _, p := range ptabs {
			ot = objw.SymPtr(s, ot, p.Linksym(), 0)
		}
		objw.Global(s, int32(ot), int16(obj.RODATA))
	}
}

func WriteImportStrings() {
	// generate import strings for imported packages
	for _, p := range types.ImportedPkgList() {
		dimportpath(p)
	}
}

func WriteBasicTypes() {
	// do basic types if compiling package runtime.
	// they have to be in at least one package,
	// and runtime is always loaded implicitly,
	// so this is as good as any.
	// another possible choice would be package main,
	// but using runtime means fewer copies in object files.
	if base.Ctxt.Pkgpath == "runtime" {
		for i := types.Kind(1); i <= types.TBOOL; i++ {
			writeType(types.NewPtr(types.Types[i]))
		}
		writeType(types.NewPtr(types.Types[types.TSTRING]))
		writeType(types.NewPtr(types.Types[types.TUNSAFEPTR]))

		// emit type structs for error and func(error) string.
		// The latter is the type of an auto-generated wrapper.
		writeType(types.NewPtr(types.ErrorType))

		writeType(types.NewSignature(types.NoPkg, nil, nil, []*types.Field{
			types.NewField(base.Pos, nil, types.ErrorType),
		}, []*types.Field{
			types.NewField(base.Pos, nil, types.Types[types.TSTRING]),
		}))

		// add paths for runtime and main, which 6l imports implicitly.
		dimportpath(ir.Pkgs.Runtime)

		if base.Flag.Race {
			dimportpath(types.NewPkg("runtime/race", ""))
		}
		if base.Flag.MSan {
			dimportpath(types.NewPkg("runtime/msan", ""))
		}

		dimportpath(types.NewPkg("main", ""))
	}
}

type typeAndStr struct {
	t       *types.Type
	short   string
	regular string
}

type typesByString []typeAndStr

func (a typesByString) Len() int { return len(a) }
func (a typesByString) Less(i, j int) bool {
	if a[i].short != a[j].short {
		return a[i].short < a[j].short
	}
	// When the only difference between the types is whether
	// they refer to byte or uint8, such as **byte vs **uint8,
	// the types' ShortStrings can be identical.
	// To preserve deterministic sort ordering, sort these by String().
	if a[i].regular != a[j].regular {
		return a[i].regular < a[j].regular
	}
	// Identical anonymous interfaces defined in different locations
	// will be equal for the above checks, but different in DWARF output.
	// Sort by source position to ensure deterministic order.
	// See issues 27013 and 30202.
	if a[i].t.Kind() == types.TINTER && a[i].t.Methods().Len() > 0 {
		return a[i].t.Methods().Index(0).Pos.Before(a[j].t.Methods().Index(0).Pos)
	}
	return false
}
func (a typesByString) Swap(i, j int) { a[i], a[j] = a[j], a[i] }

// maxPtrmaskBytes is the maximum length of a GC ptrmask bitmap,
// which holds 1-bit entries describing where pointers are in a given type.
// Above this length, the GC information is recorded as a GC program,
// which can express repetition compactly. In either form, the
// information is used by the runtime to initialize the heap bitmap,
// and for large types (like 128 or more words), they are roughly the
// same speed. GC programs are never much larger and often more
// compact. (If large arrays are involved, they can be arbitrarily
// more compact.)
//
// The cutoff must be large enough that any allocation large enough to
// use a GC program is large enough that it does not share heap bitmap
// bytes with any other objects, allowing the GC program execution to
// assume an aligned start and not use atomic operations. In the current
// runtime, this means all malloc size classes larger than the cutoff must
// be multiples of four words. On 32-bit systems that's 16 bytes, and
// all size classes >= 16 bytes are 16-byte aligned, so no real constraint.
// On 64-bit systems, that's 32 bytes, and 32-byte alignment is guaranteed
// for size classes >= 256 bytes. On a 64-bit system, 256 bytes allocated
// is 32 pointers, the bits for which fit in 4 bytes. So maxPtrmaskBytes
// must be >= 4.
//
// We used to use 16 because the GC programs do have some constant overhead
// to get started, and processing 128 pointers seems to be enough to
// amortize that overhead well.
//
// To make sure that the runtime's chansend can call typeBitsBulkBarrier,
// we raised the limit to 2048, so that even 32-bit systems are guaranteed to
// use bitmaps for objects up to 64 kB in size.
//
// Also known to reflect/type.go.
//
const maxPtrmaskBytes = 2048

// dgcsym emits and returns a data symbol containing GC information for type t,
// along with a boolean reporting whether the UseGCProg bit should be set in
// the type kind, and the ptrdata field to record in the reflect type information.
func dgcsym(t *types.Type) (lsym *obj.LSym, useGCProg bool, ptrdata int64) {
	ptrdata = types.PtrDataSize(t)
	if ptrdata/int64(types.PtrSize) <= maxPtrmaskBytes*8 {
		lsym = dgcptrmask(t)
		return
	}

	useGCProg = true
	lsym, ptrdata = dgcprog(t)
	return
}

// dgcptrmask emits and returns the symbol containing a pointer mask for type t.
func dgcptrmask(t *types.Type) *obj.LSym {
	ptrmask := make([]byte, (types.PtrDataSize(t)/int64(types.PtrSize)+7)/8)
	fillptrmask(t, ptrmask)
	p := fmt.Sprintf("gcbits.%x", ptrmask)

	sym := ir.Pkgs.Runtime.Lookup(p)
	lsym := sym.Linksym()
	if !sym.Uniq() {
		sym.SetUniq(true)
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

// dgcprog emits and returns the symbol containing a GC program for type t
// along with the size of the data described by the program (in the range
// [types.PtrDataSize(t), t.Width]).
// In practice, the size is types.PtrDataSize(t) except for non-trivial arrays.
// For non-trivial arrays, the program describes the full t.Width size.
func dgcprog(t *types.Type) (*obj.LSym, int64) {
	types.CalcSize(t)
	if t.Width == types.BADWIDTH {
		base.Fatalf("dgcprog: %v badwidth", t)
	}
	lsym := TypeLinksymPrefix(".gcprog", t)
	var p gcProg
	p.init(lsym)
	p.emit(t, 0)
	offset := p.w.BitIndex() * int64(types.PtrSize)
	p.end()
	if ptrdata := types.PtrDataSize(t); offset < ptrdata || offset > t.Width {
		base.Fatalf("dgcprog: %v: offset=%d but ptrdata=%d size=%d", t, offset, ptrdata, t.Width)
	}
	return lsym, offset
}

type gcProg struct {
	lsym   *obj.LSym
	symoff int
	w      gcprog.Writer
}

func (p *gcProg) init(lsym *obj.LSym) {
	p.lsym = lsym
	p.symoff = 4 // first 4 bytes hold program length
	p.w.Init(p.writeByte)
	if base.Debug.GCProg > 0 {
		fmt.Fprintf(os.Stderr, "compile: start GCProg for %v\n", lsym)
		p.w.Debug(os.Stderr)
	}
}

func (p *gcProg) writeByte(x byte) {
	p.symoff = objw.Uint8(p.lsym, p.symoff, x)
}

func (p *gcProg) end() {
	p.w.End()
	objw.Uint32(p.lsym, 0, uint32(p.symoff-4))
	objw.Global(p.lsym, int32(p.symoff), obj.DUPOK|obj.RODATA|obj.LOCAL)
	if base.Debug.GCProg > 0 {
		fmt.Fprintf(os.Stderr, "compile: end GCProg for %v\n", p.lsym)
	}
}

func (p *gcProg) emit(t *types.Type, offset int64) {
	types.CalcSize(t)
	if !t.HasPointers() {
		return
	}
	if t.Width == int64(types.PtrSize) {
		p.w.Ptr(offset / int64(types.PtrSize))
		return
	}
	switch t.Kind() {
	default:
		base.Fatalf("gcProg.emit: unexpected type %v", t)

	case types.TSTRING:
		p.w.Ptr(offset / int64(types.PtrSize))

	case types.TINTER:
		// Note: the first word isn't a pointer. See comment in typebits.Set
		p.w.Ptr(offset/int64(types.PtrSize) + 1)

	case types.TSLICE:
		p.w.Ptr(offset / int64(types.PtrSize))

	case types.TARRAY:
		if t.NumElem() == 0 {
			// should have been handled by haspointers check above
			base.Fatalf("gcProg.emit: empty array")
		}

		// Flatten array-of-array-of-array to just a big array by multiplying counts.
		count := t.NumElem()
		elem := t.Elem()
		for elem.IsArray() {
			count *= elem.NumElem()
			elem = elem.Elem()
		}

		if !p.w.ShouldRepeat(elem.Width/int64(types.PtrSize), count) {
			// Cheaper to just emit the bits.
			for i := int64(0); i < count; i++ {
				p.emit(elem, offset+i*elem.Width)
			}
			return
		}
		p.emit(elem, offset)
		p.w.ZeroUntil((offset + elem.Width) / int64(types.PtrSize))
		p.w.Repeat(elem.Width/int64(types.PtrSize), count-1)

	case types.TSTRUCT:
		for _, t1 := range t.Fields().Slice() {
			p.emit(t1.Type, offset+t1.Offset)
		}
	}
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
	lsym := base.PkgLinksym("go.map", "zero", obj.ABI0)
	x := ir.NewLinksymExpr(base.Pos, lsym, types.Types[types.TUINT8])
	return typecheck.Expr(typecheck.NodAddr(x))
}

func CollectPTabs() {
	if !base.Ctxt.Flag_dynlink || types.LocalPkg.Name != "main" {
		return
	}
	for _, exportn := range typecheck.Target.Exports {
		s := exportn.Sym()
		nn := ir.AsNode(s.Def)
		if nn == nil {
			continue
		}
		if nn.Op() != ir.ONAME {
			continue
		}
		n := nn.(*ir.Name)
		if !types.IsExported(s.Name) {
			continue
		}
		if s.Pkg.Name != "main" {
			continue
		}
		ptabs = append(ptabs, n)
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
func methodWrapper(rcvr *types.Type, method *types.Field) *obj.LSym {
	newnam := ir.MethodSym(rcvr, method.Sym)
	lsym := newnam.Linksym()
	if newnam.Siggen() {
		return lsym
	}
	newnam.SetSiggen(true)

	if types.Identical(rcvr, method.Type.Recv().Type) {
		return lsym
	}

	// Only generate (*T).M wrappers for T.M in T's own package.
	if rcvr.IsPtr() && rcvr.Elem() == method.Type.Recv().Type &&
		rcvr.Elem().Sym() != nil && rcvr.Elem().Sym().Pkg != types.LocalPkg {
		return lsym
	}

	// Only generate I.M wrappers for I in I's own package
	// but keep doing it for error.Error (was issue #29304).
	if rcvr.IsInterface() && rcvr.Sym() != nil && rcvr.Sym().Pkg != types.LocalPkg && rcvr != types.ErrorType {
		return lsym
	}

	base.Pos = base.AutogeneratedPos
	typecheck.DeclContext = ir.PEXTERN

	tfn := ir.NewFuncType(base.Pos,
		ir.NewField(base.Pos, typecheck.Lookup(".this"), nil, rcvr),
		typecheck.NewFuncParams(method.Type.Params(), true),
		typecheck.NewFuncParams(method.Type.Results(), false))

	// TODO(austin): SelectorExpr may have created one or more
	// ir.Names for these already with a nil Func field. We should
	// consolidate these and always attach a Func to the Name.
	fn := typecheck.DeclFunc(newnam, tfn)
	fn.SetDupok(true)

	nthis := ir.AsNode(tfn.Type().Recv().Nname)

	methodrcvr := method.Type.Recv().Type

	// generate nil pointer check for better error
	if rcvr.IsPtr() && rcvr.Elem() == methodrcvr {
		// generating wrapper from *T to T.
		n := ir.NewIfStmt(base.Pos, nil, nil, nil)
		n.Cond = ir.NewBinaryExpr(base.Pos, ir.OEQ, nthis, typecheck.NodNil())
		call := ir.NewCallExpr(base.Pos, ir.OCALL, typecheck.LookupRuntime("panicwrap"), nil)
		n.Body = []ir.Node{call}
		fn.Body.Append(n)
	}

	dot := typecheck.AddImplicitDots(ir.NewSelectorExpr(base.Pos, ir.OXDOT, nthis, method.Sym))

	// generate call
	// It's not possible to use a tail call when dynamic linking on ppc64le. The
	// bad scenario is when a local call is made to the wrapper: the wrapper will
	// call the implementation, which might be in a different module and so set
	// the TOC to the appropriate value for that module. But if it returns
	// directly to the wrapper's caller, nothing will reset it to the correct
	// value for that function.
	//
	// Disable tailcall for RegabiArgs for now. The IR does not connect the
	// arguments with the OTAILCALL node, and the arguments are not marshaled
	// correctly.
	if !base.Flag.Cfg.Instrumenting && rcvr.IsPtr() && methodrcvr.IsPtr() && method.Embedded != 0 && !types.IsInterfaceMethod(method.Type) && !(base.Ctxt.Arch.Name == "ppc64le" && base.Ctxt.Flag_dynlink) && !objabi.Experiment.RegabiArgs {
		// generate tail call: adjust pointer receiver and jump to embedded method.
		left := dot.X // skip final .M
		if !left.Type().IsPtr() {
			left = typecheck.NodAddr(left)
		}
		as := ir.NewAssignStmt(base.Pos, nthis, typecheck.ConvNop(left, rcvr))
		fn.Body.Append(as)
		fn.Body.Append(ir.NewTailCallStmt(base.Pos, method.Nname.(*ir.Name)))
	} else {
		fn.SetWrapper(true) // ignore frame for panic+recover matching
		call := ir.NewCallExpr(base.Pos, ir.OCALL, dot, nil)
		call.Args = ir.ParamNames(tfn.Type())
		call.IsDDD = tfn.Type().IsVariadic()
		if method.Type.NumResults() > 0 {
			ret := ir.NewReturnStmt(base.Pos, nil)
			ret.Results = []ir.Node{call}
			fn.Body.Append(ret)
		} else {
			fn.Body.Append(call)
		}
	}

	typecheck.FinishFuncBody()
	if base.Debug.DclStack != 0 {
		types.CheckDclstack()
	}

	typecheck.Func(fn)
	ir.CurFunc = fn
	typecheck.Stmts(fn.Body)

	// Inline calls within (*T).M wrappers. This is safe because we only
	// generate those wrappers within the same compilation unit as (T).M.
	// TODO(mdempsky): Investigate why we can't enable this more generally.
	if rcvr.IsPtr() && rcvr.Elem() == method.Type.Recv().Type && rcvr.Elem().Sym() != nil {
		inline.InlineCalls(fn)
	}
	escape.Batch([]*ir.Func{fn}, false)

	ir.CurFunc = nil
	typecheck.Target.Decls = append(typecheck.Target.Decls, fn)

	return lsym
}

var ZeroSize int64

// MarkTypeUsedInInterface marks that type t is converted to an interface.
// This information is used in the linker in dead method elimination.
func MarkTypeUsedInInterface(t *types.Type, from *obj.LSym) {
	tsym := TypeLinksym(t)
	// Emit a marker relocation. The linker will know the type is converted
	// to an interface if "from" is reachable.
	r := obj.Addrel(from)
	r.Sym = tsym
	r.Type = objabi.R_USEIFACE
}

// MarkUsedIfaceMethod marks that an interface method is used in the current
// function. n is OCALLINTER node.
func MarkUsedIfaceMethod(n *ir.CallExpr) {
	dot := n.X.(*ir.SelectorExpr)
	ityp := dot.X.Type()
	tsym := TypeLinksym(ityp)
	r := obj.Addrel(ir.CurFunc.LSym)
	r.Sym = tsym
	// dot.Xoffset is the method index * PtrSize (the offset of code pointer
	// in itab).
	midx := dot.Offset() / int64(types.PtrSize)
	r.Add = InterfaceMethodOffset(ityp, midx)
	r.Type = objabi.R_USEIFACEMETHOD
}

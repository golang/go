// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/compile/internal/types"
	"cmd/internal/gcprog"
	"cmd/internal/obj"
	"cmd/internal/objabi"
	"cmd/internal/src"
	"fmt"
	"os"
	"sort"
	"strings"
	"sync"
)

type itabEntry struct {
	t, itype *types.Type
	lsym     *obj.LSym // symbol of the itab itself

	// symbols of each method in
	// the itab, sorted by byte offset;
	// filled in by peekitabs
	entries []*obj.LSym
}

type ptabEntry struct {
	s *types.Sym
	t *types.Type
}

// runtime interface and reflection data structures
var (
	signatsetmu sync.Mutex // protects signatset
	signatset   = make(map[*types.Type]struct{})

	itabs []itabEntry
	ptabs []ptabEntry
)

type Sig struct {
	name   string
	pkg    *types.Pkg
	isym   *types.Sym
	tsym   *types.Sym
	type_  *types.Type
	mtype  *types.Type
	offset int32
}

// siglt sorts method signatures by name, then package path.
func siglt(a, b *Sig) bool {
	if a.name != b.name {
		return a.name < b.name
	}
	if a.pkg == b.pkg {
		return false
	}
	if a.pkg == nil {
		return true
	}
	if b.pkg == nil {
		return false
	}
	return a.pkg.Path < b.pkg.Path
}

// Builds a type representing a Bucket structure for
// the given map type. This type is not visible to users -
// we include only enough information to generate a correct GC
// program for it.
// Make sure this stays in sync with ../../../../runtime/hashmap.go!
const (
	BUCKETSIZE = 8
	MAXKEYSIZE = 128
	MAXVALSIZE = 128
)

func structfieldSize() int { return 3 * Widthptr } // Sizeof(runtime.structfield{})
func imethodSize() int     { return 4 + 4 }        // Sizeof(runtime.imethod{})
func uncommonSize(t *types.Type) int { // Sizeof(runtime.uncommontype{})
	if t.Sym == nil && len(methods(t)) == 0 {
		return 0
	}
	return 4 + 2 + 2 + 4 + 4
}

func makefield(name string, t *types.Type) *types.Field {
	f := types.NewField()
	f.Type = t
	f.Sym = (*types.Pkg)(nil).Lookup(name)
	return f
}

func mapbucket(t *types.Type) *types.Type {
	if t.MapType().Bucket != nil {
		return t.MapType().Bucket
	}

	bucket := types.New(TSTRUCT)
	keytype := t.Key()
	valtype := t.Val()
	dowidth(keytype)
	dowidth(valtype)
	if keytype.Width > MAXKEYSIZE {
		keytype = types.NewPtr(keytype)
	}
	if valtype.Width > MAXVALSIZE {
		valtype = types.NewPtr(valtype)
	}

	field := make([]*types.Field, 0, 5)

	// The first field is: uint8 topbits[BUCKETSIZE].
	arr := types.NewArray(types.Types[TUINT8], BUCKETSIZE)
	field = append(field, makefield("topbits", arr))

	arr = types.NewArray(keytype, BUCKETSIZE)
	arr.SetNoalg(true)
	field = append(field, makefield("keys", arr))

	arr = types.NewArray(valtype, BUCKETSIZE)
	arr.SetNoalg(true)
	field = append(field, makefield("values", arr))

	// Make sure the overflow pointer is the last memory in the struct,
	// because the runtime assumes it can use size-ptrSize as the
	// offset of the overflow pointer. We double-check that property
	// below once the offsets and size are computed.
	//
	// BUCKETSIZE is 8, so the struct is aligned to 64 bits to this point.
	// On 32-bit systems, the max alignment is 32-bit, and the
	// overflow pointer will add another 32-bit field, and the struct
	// will end with no padding.
	// On 64-bit systems, the max alignment is 64-bit, and the
	// overflow pointer will add another 64-bit field, and the struct
	// will end with no padding.
	// On nacl/amd64p32, however, the max alignment is 64-bit,
	// but the overflow pointer will add only a 32-bit field,
	// so if the struct needs 64-bit padding (because a key or value does)
	// then it would end with an extra 32-bit padding field.
	// Preempt that by emitting the padding here.
	if int(t.Val().Align) > Widthptr || int(t.Key().Align) > Widthptr {
		field = append(field, makefield("pad", types.Types[TUINTPTR]))
	}

	// If keys and values have no pointers, the map implementation
	// can keep a list of overflow pointers on the side so that
	// buckets can be marked as having no pointers.
	// Arrange for the bucket to have no pointers by changing
	// the type of the overflow field to uintptr in this case.
	// See comment on hmap.overflow in ../../../../runtime/hashmap.go.
	otyp := types.NewPtr(bucket)
	if !types.Haspointers(t.Val()) && !types.Haspointers(t.Key()) && t.Val().Width <= MAXVALSIZE && t.Key().Width <= MAXKEYSIZE {
		otyp = types.Types[TUINTPTR]
	}
	ovf := makefield("overflow", otyp)
	field = append(field, ovf)

	// link up fields
	bucket.SetNoalg(true)
	bucket.SetLocal(t.Local())
	bucket.SetFields(field[:])
	dowidth(bucket)

	// Double-check that overflow field is final memory in struct,
	// with no padding at end. See comment above.
	if ovf.Offset != bucket.Width-int64(Widthptr) {
		Fatalf("bad math in mapbucket for %v", t)
	}

	t.MapType().Bucket = bucket

	bucket.StructType().Map = t
	return bucket
}

// Builds a type representing a Hmap structure for the given map type.
// Make sure this stays in sync with ../../../../runtime/hashmap.go!
func hmap(t *types.Type) *types.Type {
	if t.MapType().Hmap != nil {
		return t.MapType().Hmap
	}

	bucket := mapbucket(t)
	fields := []*types.Field{
		makefield("count", types.Types[TINT]),
		makefield("flags", types.Types[TUINT8]),
		makefield("B", types.Types[TUINT8]),
		makefield("noverflow", types.Types[TUINT16]),
		makefield("hash0", types.Types[TUINT32]),
		makefield("buckets", types.NewPtr(bucket)),
		makefield("oldbuckets", types.NewPtr(bucket)),
		makefield("nevacuate", types.Types[TUINTPTR]),
		makefield("overflow", types.Types[TUNSAFEPTR]),
	}

	h := types.New(TSTRUCT)
	h.SetNoalg(true)
	h.SetLocal(t.Local())
	h.SetFields(fields)
	dowidth(h)
	t.MapType().Hmap = h
	h.StructType().Map = t
	return h
}

func hiter(t *types.Type) *types.Type {
	if t.MapType().Hiter != nil {
		return t.MapType().Hiter
	}

	// build a struct:
	// hiter {
	//    key *Key
	//    val *Value
	//    t *MapType
	//    h *Hmap
	//    buckets *Bucket
	//    bptr *Bucket
	//    overflow0 unsafe.Pointer
	//    overflow1 unsafe.Pointer
	//    startBucket uintptr
	//    stuff uintptr
	//    bucket uintptr
	//    checkBucket uintptr
	// }
	// must match ../../../../runtime/hashmap.go:hiter.
	var field [12]*types.Field
	field[0] = makefield("key", types.NewPtr(t.Key()))
	field[1] = makefield("val", types.NewPtr(t.Val()))
	field[2] = makefield("t", types.NewPtr(types.Types[TUINT8]))
	field[3] = makefield("h", types.NewPtr(hmap(t)))
	field[4] = makefield("buckets", types.NewPtr(mapbucket(t)))
	field[5] = makefield("bptr", types.NewPtr(mapbucket(t)))
	field[6] = makefield("overflow0", types.Types[TUNSAFEPTR])
	field[7] = makefield("overflow1", types.Types[TUNSAFEPTR])
	field[8] = makefield("startBucket", types.Types[TUINTPTR])
	field[9] = makefield("stuff", types.Types[TUINTPTR]) // offset+wrapped+B+I
	field[10] = makefield("bucket", types.Types[TUINTPTR])
	field[11] = makefield("checkBucket", types.Types[TUINTPTR])

	// build iterator struct holding the above fields
	i := types.New(TSTRUCT)
	i.SetNoalg(true)
	i.SetFields(field[:])
	dowidth(i)
	if i.Width != int64(12*Widthptr) {
		Fatalf("hash_iter size not correct %d %d", i.Width, 12*Widthptr)
	}
	t.MapType().Hiter = i
	i.StructType().Map = t
	return i
}

// f is method type, with receiver.
// return function type, receiver as first argument (or not).
func methodfunc(f *types.Type, receiver *types.Type) *types.Type {
	var in []*Node
	if receiver != nil {
		d := nod(ODCLFIELD, nil, nil)
		d.Type = receiver
		in = append(in, d)
	}

	var d *Node
	for _, t := range f.Params().Fields().Slice() {
		d = nod(ODCLFIELD, nil, nil)
		d.Type = t.Type
		d.SetIsddd(t.Isddd())
		in = append(in, d)
	}

	var out []*Node
	for _, t := range f.Results().Fields().Slice() {
		d = nod(ODCLFIELD, nil, nil)
		d.Type = t.Type
		out = append(out, d)
	}

	t := functype(nil, in, out)
	if f.Nname() != nil {
		// Link to name of original method function.
		t.SetNname(f.Nname())
	}

	return t
}

// methods returns the methods of the non-interface type t, sorted by name.
// Generates stub functions as needed.
func methods(t *types.Type) []*Sig {
	// method type
	mt := methtype(t)

	if mt == nil {
		return nil
	}
	expandmeth(mt)

	// type stored in interface word
	it := t

	if !isdirectiface(it) {
		it = types.NewPtr(t)
	}

	// make list of methods for t,
	// generating code if necessary.
	var ms []*Sig
	for _, f := range mt.AllMethods().Slice() {
		if f.Type.Etype != TFUNC || f.Type.Recv() == nil {
			Fatalf("non-method on %v method %v %v\n", mt, f.Sym, f)
		}
		if f.Type.Recv() == nil {
			Fatalf("receiver with no type on %v method %v %v\n", mt, f.Sym, f)
		}
		if f.Nointerface() {
			continue
		}

		method := f.Sym
		if method == nil {
			continue
		}

		// get receiver type for this particular method.
		// if pointer receiver but non-pointer t and
		// this is not an embedded pointer inside a struct,
		// method does not apply.
		this := f.Type.Recv().Type

		if this.IsPtr() && this.Elem() == t {
			continue
		}
		if this.IsPtr() && !t.IsPtr() && f.Embedded != 2 && !isifacemethod(f.Type) {
			continue
		}

		var sig Sig
		ms = append(ms, &sig)

		sig.name = method.Name
		if !exportname(method.Name) {
			if method.Pkg == nil {
				Fatalf("methods: missing package")
			}
			sig.pkg = method.Pkg
		}

		sig.isym = methodsym(method, it, true)
		sig.tsym = methodsym(method, t, false)
		sig.type_ = methodfunc(f.Type, t)
		sig.mtype = methodfunc(f.Type, nil)

		if !sig.isym.Siggen() {
			sig.isym.SetSiggen(true)
			if !eqtype(this, it) || this.Width < int64(Widthptr) {
				compiling_wrappers = 1
				genwrapper(it, f, sig.isym, 1)
				compiling_wrappers = 0
			}
		}

		if !sig.tsym.Siggen() {
			sig.tsym.SetSiggen(true)
			if !eqtype(this, t) {
				compiling_wrappers = 1
				genwrapper(t, f, sig.tsym, 0)
				compiling_wrappers = 0
			}
		}
	}

	obj.SortSlice(ms, func(i, j int) bool { return siglt(ms[i], ms[j]) })
	return ms
}

// imethods returns the methods of the interface type t, sorted by name.
func imethods(t *types.Type) []*Sig {
	var methods []*Sig
	for _, f := range t.Fields().Slice() {
		if f.Type.Etype != TFUNC || f.Sym == nil {
			continue
		}
		method := f.Sym
		var sig = Sig{
			name: method.Name,
		}
		if !exportname(method.Name) {
			if method.Pkg == nil {
				Fatalf("imethods: missing package")
			}
			sig.pkg = method.Pkg
		}

		sig.mtype = f.Type
		sig.offset = 0
		sig.type_ = methodfunc(f.Type, nil)

		if n := len(methods); n > 0 {
			last := methods[n-1]
			if !(siglt(last, &sig)) {
				Fatalf("sigcmp vs sortinter %s %s", last.name, sig.name)
			}
		}
		methods = append(methods, &sig)

		// Compiler can only refer to wrappers for non-blank methods.
		if method.IsBlank() {
			continue
		}

		// NOTE(rsc): Perhaps an oversight that
		// IfaceType.Method is not in the reflect data.
		// Generate the method body, so that compiled
		// code can refer to it.
		isym := methodsym(method, t, false)
		if !isym.Siggen() {
			isym.SetSiggen(true)
			genwrapper(t, f, isym, 0)
		}
	}

	return methods
}

func dimportpath(p *types.Pkg) {
	if p.Pathsym != nil {
		return
	}

	// If we are compiling the runtime package, there are two runtime packages around
	// -- localpkg and Runtimepkg. We don't want to produce import path symbols for
	// both of them, so just produce one for localpkg.
	if myimportpath == "runtime" && p == Runtimepkg {
		return
	}

	var str string
	if p == localpkg {
		// Note: myimportpath != "", or else dgopkgpath won't call dimportpath.
		str = myimportpath
	} else {
		str = p.Path
	}

	s := Ctxt.Lookup("type..importpath." + p.Prefix + ".")
	ot := dnameData(s, 0, str, "", nil, false)
	ggloblsym(s, int32(ot), obj.DUPOK|obj.RODATA)
	p.Pathsym = s
}

func dgopkgpath(s *obj.LSym, ot int, pkg *types.Pkg) int {
	if pkg == nil {
		return duintptr(s, ot, 0)
	}

	if pkg == localpkg && myimportpath == "" {
		// If we don't know the full import path of the package being compiled
		// (i.e. -p was not passed on the compiler command line), emit a reference to
		// type..importpath.""., which the linker will rewrite using the correct import path.
		// Every package that imports this one directly defines the symbol.
		// See also https://groups.google.com/forum/#!topic/golang-dev/myb9s53HxGQ.
		ns := Ctxt.Lookup(`type..importpath."".`)
		return dsymptr(s, ot, ns, 0)
	}

	dimportpath(pkg)
	return dsymptr(s, ot, pkg.Pathsym, 0)
}

// dgopkgpathOff writes an offset relocation in s at offset ot to the pkg path symbol.
func dgopkgpathOff(s *obj.LSym, ot int, pkg *types.Pkg) int {
	if pkg == nil {
		return duint32(s, ot, 0)
	}
	if pkg == localpkg && myimportpath == "" {
		// If we don't know the full import path of the package being compiled
		// (i.e. -p was not passed on the compiler command line), emit a reference to
		// type..importpath.""., which the linker will rewrite using the correct import path.
		// Every package that imports this one directly defines the symbol.
		// See also https://groups.google.com/forum/#!topic/golang-dev/myb9s53HxGQ.
		ns := Ctxt.Lookup(`type..importpath."".`)
		return dsymptrOff(s, ot, ns, 0)
	}

	dimportpath(pkg)
	return dsymptrOff(s, ot, pkg.Pathsym, 0)
}

// isExportedField reports whether a struct field is exported.
// It also returns the package to use for PkgPath for an unexported field.
func isExportedField(ft *types.Field) (bool, *types.Pkg) {
	if ft.Sym != nil && ft.Embedded == 0 {
		return exportname(ft.Sym.Name), ft.Sym.Pkg
	} else {
		if ft.Type.Sym != nil &&
			(ft.Type.Sym.Pkg == builtinpkg || !exportname(ft.Type.Sym.Name)) {
			return false, ft.Type.Sym.Pkg
		} else {
			return true, nil
		}
	}
}

// dnameField dumps a reflect.name for a struct field.
func dnameField(lsym *obj.LSym, ot int, spkg *types.Pkg, ft *types.Field) int {
	var name string
	if ft.Sym != nil {
		name = ft.Sym.Name
	}
	isExported, fpkg := isExportedField(ft)
	if isExported || fpkg == spkg {
		fpkg = nil
	}
	nsym := dname(name, ft.Note, fpkg, isExported)
	return dsymptr(lsym, ot, nsym, 0)
}

// dnameData writes the contents of a reflect.name into s at offset ot.
func dnameData(s *obj.LSym, ot int, name, tag string, pkg *types.Pkg, exported bool) int {
	if len(name) > 1<<16-1 {
		Fatalf("name too long: %s", name)
	}
	if len(tag) > 1<<16-1 {
		Fatalf("tag too long: %s", tag)
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

	ot = int(s.WriteBytes(Ctxt, int64(ot), b))

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
	s := Ctxt.Lookup(sname)
	if len(s.P) > 0 {
		return s
	}
	ot := dnameData(s, 0, name, tag, pkg, exported)
	ggloblsym(s, int32(ot), obj.DUPOK|obj.RODATA)
	return s
}

// dextratype dumps the fields of a runtime.uncommontype.
// dataAdd is the offset in bytes after the header where the
// backing array of the []method field is written (by dextratypeData).
func dextratype(lsym *obj.LSym, ot int, t *types.Type, dataAdd int) int {
	m := methods(t)
	if t.Sym == nil && len(m) == 0 {
		return ot
	}
	noff := int(Rnd(int64(ot), int64(Widthptr)))
	if noff != ot {
		Fatalf("unexpected alignment in dextratype for %v", t)
	}

	for _, a := range m {
		dtypesym(a.type_)
	}

	ot = dgopkgpathOff(lsym, ot, typePkg(t))

	dataAdd += uncommonSize(t)
	mcount := len(m)
	if mcount != int(uint16(mcount)) {
		Fatalf("too many methods on %v: %d", t, mcount)
	}
	if dataAdd != int(uint32(dataAdd)) {
		Fatalf("methods are too far away on %v: %d", t, dataAdd)
	}

	ot = duint16(lsym, ot, uint16(mcount))
	ot = duint16(lsym, ot, 0)
	ot = duint32(lsym, ot, uint32(dataAdd))
	ot = duint32(lsym, ot, 0)
	return ot
}

func typePkg(t *types.Type) *types.Pkg {
	tsym := t.Sym
	if tsym == nil {
		switch t.Etype {
		case TARRAY, TSLICE, TPTR32, TPTR64, TCHAN:
			if t.Elem() != nil {
				tsym = t.Elem().Sym
			}
		}
	}
	if tsym != nil && t != types.Types[t.Etype] && t != types.Errortype {
		return tsym.Pkg
	}
	return nil
}

// dextratypeData dumps the backing array for the []method field of
// runtime.uncommontype.
func dextratypeData(lsym *obj.LSym, ot int, t *types.Type) int {
	for _, a := range methods(t) {
		// ../../../../runtime/type.go:/method
		exported := exportname(a.name)
		var pkg *types.Pkg
		if !exported && a.pkg != typePkg(t) {
			pkg = a.pkg
		}
		nsym := dname(a.name, "", pkg, exported)

		ot = dsymptrOff(lsym, ot, nsym, 0)
		ot = dmethodptrOff(lsym, ot, dtypesym(a.mtype).Linksym())
		ot = dmethodptrOff(lsym, ot, a.isym.Linksym())
		ot = dmethodptrOff(lsym, ot, a.tsym.Linksym())
	}
	return ot
}

func dmethodptrOff(s *obj.LSym, ot int, x *obj.LSym) int {
	duint32(s, ot, 0)
	r := obj.Addrel(s)
	r.Off = int32(ot)
	r.Siz = 4
	r.Sym = x
	r.Type = objabi.R_METHODOFF
	return ot + 4
}

var kinds = []int{
	TINT:        objabi.KindInt,
	TUINT:       objabi.KindUint,
	TINT8:       objabi.KindInt8,
	TUINT8:      objabi.KindUint8,
	TINT16:      objabi.KindInt16,
	TUINT16:     objabi.KindUint16,
	TINT32:      objabi.KindInt32,
	TUINT32:     objabi.KindUint32,
	TINT64:      objabi.KindInt64,
	TUINT64:     objabi.KindUint64,
	TUINTPTR:    objabi.KindUintptr,
	TFLOAT32:    objabi.KindFloat32,
	TFLOAT64:    objabi.KindFloat64,
	TBOOL:       objabi.KindBool,
	TSTRING:     objabi.KindString,
	TPTR32:      objabi.KindPtr,
	TPTR64:      objabi.KindPtr,
	TSTRUCT:     objabi.KindStruct,
	TINTER:      objabi.KindInterface,
	TCHAN:       objabi.KindChan,
	TMAP:        objabi.KindMap,
	TARRAY:      objabi.KindArray,
	TSLICE:      objabi.KindSlice,
	TFUNC:       objabi.KindFunc,
	TCOMPLEX64:  objabi.KindComplex64,
	TCOMPLEX128: objabi.KindComplex128,
	TUNSAFEPTR:  objabi.KindUnsafePointer,
}

// typeptrdata returns the length in bytes of the prefix of t
// containing pointer data. Anything after this offset is scalar data.
func typeptrdata(t *types.Type) int64 {
	if !types.Haspointers(t) {
		return 0
	}

	switch t.Etype {
	case TPTR32,
		TPTR64,
		TUNSAFEPTR,
		TFUNC,
		TCHAN,
		TMAP:
		return int64(Widthptr)

	case TSTRING:
		// struct { byte *str; intgo len; }
		return int64(Widthptr)

	case TINTER:
		// struct { Itab *tab;	void *data; } or
		// struct { Type *type; void *data; }
		return 2 * int64(Widthptr)

	case TSLICE:
		// struct { byte *array; uintgo len; uintgo cap; }
		return int64(Widthptr)

	case TARRAY:
		// haspointers already eliminated t.NumElem() == 0.
		return (t.NumElem()-1)*t.Elem().Width + typeptrdata(t.Elem())

	case TSTRUCT:
		// Find the last field that has pointers.
		var lastPtrField *types.Field
		for _, t1 := range t.Fields().Slice() {
			if types.Haspointers(t1.Type) {
				lastPtrField = t1
			}
		}
		return lastPtrField.Offset + typeptrdata(lastPtrField.Type)

	default:
		Fatalf("typeptrdata: unexpected type, %v", t)
		return 0
	}
}

// tflag is documented in reflect/type.go.
//
// tflag values must be kept in sync with copies in:
//	cmd/compile/internal/gc/reflect.go
//	cmd/link/internal/ld/decodesym.go
//	reflect/type.go
//	runtime/type.go
const (
	tflagUncommon  = 1 << 0
	tflagExtraStar = 1 << 1
	tflagNamed     = 1 << 2
)

var (
	algarray       *obj.LSym
	memhashvarlen  *obj.LSym
	memequalvarlen *obj.LSym
)

// dcommontype dumps the contents of a reflect.rtype (runtime._type).
func dcommontype(lsym *obj.LSym, ot int, t *types.Type) int {
	if ot != 0 {
		Fatalf("dcommontype %d", ot)
	}

	sizeofAlg := 2 * Widthptr
	if algarray == nil {
		algarray = Sysfunc("algarray")
	}
	dowidth(t)
	alg := algtype(t)
	var algsym *obj.LSym
	if alg == ASPECIAL || alg == AMEM {
		algsym = dalgsym(t)
	}

	sptrWeak := true
	var sptr *obj.LSym
	if !t.IsPtr() || t.PtrBase != nil {
		tptr := types.NewPtr(t)
		if t.Sym != nil || methods(tptr) != nil {
			sptrWeak = false
		}
		sptr = dtypesym(tptr).Linksym()
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
	//		alg           *typeAlg
	//		gcdata        *byte
	//		str           nameOff
	//		ptrToThis     typeOff
	//	}
	ot = duintptr(lsym, ot, uint64(t.Width))
	ot = duintptr(lsym, ot, uint64(ptrdata))
	ot = duint32(lsym, ot, typehash(t))

	var tflag uint8
	if uncommonSize(t) != 0 {
		tflag |= tflagUncommon
	}
	if t.Sym != nil && t.Sym.Name != "" {
		tflag |= tflagNamed
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
		if t.Sym != nil {
			exported = exportname(t.Sym.Name)
		}
	} else {
		if t.Elem() != nil && t.Elem().Sym != nil {
			exported = exportname(t.Elem().Sym.Name)
		}
	}

	ot = duint8(lsym, ot, tflag)

	// runtime (and common sense) expects alignment to be a power of two.
	i := int(t.Align)

	if i == 0 {
		i = 1
	}
	if i&(i-1) != 0 {
		Fatalf("invalid alignment %d for %v", t.Align, t)
	}
	ot = duint8(lsym, ot, t.Align) // align
	ot = duint8(lsym, ot, t.Align) // fieldAlign

	i = kinds[t.Etype]
	if !types.Haspointers(t) {
		i |= objabi.KindNoPointers
	}
	if isdirectiface(t) {
		i |= objabi.KindDirectIface
	}
	if useGCProg {
		i |= objabi.KindGCProg
	}
	ot = duint8(lsym, ot, uint8(i)) // kind
	if algsym == nil {
		ot = dsymptr(lsym, ot, algarray, int(alg)*sizeofAlg)
	} else {
		ot = dsymptr(lsym, ot, algsym, 0)
	}
	ot = dsymptr(lsym, ot, gcsym, 0) // gcdata

	nsym := dname(p, "", nil, exported)
	ot = dsymptrOff(lsym, ot, nsym, 0) // str
	// ptrToThis
	if sptr == nil {
		ot = duint32(lsym, ot, 0)
	} else if sptrWeak {
		ot = dsymptrWeakOff(lsym, ot, sptr)
	} else {
		ot = dsymptrOff(lsym, ot, sptr, 0)
	}

	return ot
}

func typesymname(t *types.Type) string {
	name := t.ShortString()
	// Use a separate symbol name for Noalg types for #17752.
	if a, bad := algtype1(t); a == ANOEQ && bad.Noalg() {
		name = "noalg." + name
	}
	return name
}

// Fake package for runtime type info (headers)
// Don't access directly, use typeLookup below.
var (
	typepkgmu sync.Mutex // protects typepkg lookups
	typepkg   = types.NewPkg("type", "type")
)

func typeLookup(name string) *types.Sym {
	typepkgmu.Lock()
	s := typepkg.Lookup(name)
	typepkgmu.Unlock()
	return s
}

func typesym(t *types.Type) *types.Sym {
	return typeLookup(typesymname(t))
}

// tracksym returns the symbol for tracking use of field/method f, assumed
// to be a member of struct/interface type t.
func tracksym(t *types.Type, f *types.Field) *types.Sym {
	return trackpkg.Lookup(t.ShortString() + "." + f.Sym.Name)
}

func typesymprefix(prefix string, t *types.Type) *types.Sym {
	p := prefix + "." + t.ShortString()
	s := typeLookup(p)

	//print("algsym: %s -> %+S\n", p, s);

	return s
}

func typenamesym(t *types.Type) *types.Sym {
	if t == nil || (t.IsPtr() && t.Elem() == nil) || t.IsUntyped() {
		Fatalf("typenamesym %v", t)
	}
	s := typesym(t)
	signatsetmu.Lock()
	addsignat(t)
	signatsetmu.Unlock()
	return s
}

func typename(t *types.Type) *Node {
	s := typenamesym(t)
	if s.Def == nil {
		n := newnamel(src.NoXPos, s)
		n.Type = types.Types[TUINT8]
		n.SetClass(PEXTERN)
		n.SetTypecheck(1)
		s.Def = asTypesNode(n)
	}

	n := nod(OADDR, asNode(s.Def), nil)
	n.Type = types.NewPtr(asNode(s.Def).Type)
	n.SetAddable(true)
	n.SetTypecheck(1)
	return n
}

func itabname(t, itype *types.Type) *Node {
	if t == nil || (t.IsPtr() && t.Elem() == nil) || t.IsUntyped() || !itype.IsInterface() || itype.IsEmptyInterface() {
		Fatalf("itabname(%v, %v)", t, itype)
	}
	s := itabpkg.Lookup(t.ShortString() + "," + itype.ShortString())
	if s.Def == nil {
		n := newname(s)
		n.Type = types.Types[TUINT8]
		n.SetClass(PEXTERN)
		n.SetTypecheck(1)
		s.Def = asTypesNode(n)
		itabs = append(itabs, itabEntry{t: t, itype: itype, lsym: s.Linksym()})
	}

	n := nod(OADDR, asNode(s.Def), nil)
	n.Type = types.NewPtr(asNode(s.Def).Type)
	n.SetAddable(true)
	n.SetTypecheck(1)
	return n
}

// isreflexive reports whether t has a reflexive equality operator.
// That is, if x==x for all x of type t.
func isreflexive(t *types.Type) bool {
	switch t.Etype {
	case TBOOL,
		TINT,
		TUINT,
		TINT8,
		TUINT8,
		TINT16,
		TUINT16,
		TINT32,
		TUINT32,
		TINT64,
		TUINT64,
		TUINTPTR,
		TPTR32,
		TPTR64,
		TUNSAFEPTR,
		TSTRING,
		TCHAN:
		return true

	case TFLOAT32,
		TFLOAT64,
		TCOMPLEX64,
		TCOMPLEX128,
		TINTER:
		return false

	case TARRAY:
		return isreflexive(t.Elem())

	case TSTRUCT:
		for _, t1 := range t.Fields().Slice() {
			if !isreflexive(t1.Type) {
				return false
			}
		}
		return true

	default:
		Fatalf("bad type for map key: %v", t)
		return false
	}
}

// needkeyupdate reports whether map updates with t as a key
// need the key to be updated.
func needkeyupdate(t *types.Type) bool {
	switch t.Etype {
	case TBOOL, TINT, TUINT, TINT8, TUINT8, TINT16, TUINT16, TINT32, TUINT32,
		TINT64, TUINT64, TUINTPTR, TPTR32, TPTR64, TUNSAFEPTR, TCHAN:
		return false

	case TFLOAT32, TFLOAT64, TCOMPLEX64, TCOMPLEX128, // floats and complex can be +0/-0
		TINTER,
		TSTRING: // strings might have smaller backing stores
		return true

	case TARRAY:
		return needkeyupdate(t.Elem())

	case TSTRUCT:
		for _, t1 := range t.Fields().Slice() {
			if needkeyupdate(t1.Type) {
				return true
			}
		}
		return false

	default:
		Fatalf("bad type for map key: %v", t)
		return true
	}
}

// formalType replaces byte and rune aliases with real types.
// They've been separate internally to make error messages
// better, but we have to merge them in the reflect tables.
func formalType(t *types.Type) *types.Type {
	if t == types.Bytetype || t == types.Runetype {
		return types.Types[t.Etype]
	}
	return t
}

func dtypesym(t *types.Type) *types.Sym {
	t = formalType(t)
	if t.IsUntyped() {
		Fatalf("dtypesym %v", t)
	}

	s := typesym(t)
	if s.Siggen() {
		return s
	}
	s.SetSiggen(true)

	// special case (look for runtime below):
	// when compiling package runtime,
	// emit the type structures for int, float, etc.
	tbase := t

	if t.IsPtr() && t.Sym == nil && t.Elem().Sym != nil {
		tbase = t.Elem()
	}
	dupok := 0
	if tbase.Sym == nil {
		dupok = obj.DUPOK
	}

	if myimportpath == "runtime" && (tbase == types.Types[tbase.Etype] || tbase == types.Bytetype || tbase == types.Runetype || tbase == types.Errortype) { // int, float, etc
		goto ok
	}

	// named types from other files are defined only by those files
	if tbase.Sym != nil && !tbase.Local() {
		return s
	}
	if isforw[tbase.Etype] {
		return s
	}

ok:
	ot := 0
	lsym := s.Linksym()
	switch t.Etype {
	default:
		ot = dcommontype(lsym, ot, t)
		ot = dextratype(lsym, ot, t, 0)

	case TARRAY:
		// ../../../../runtime/type.go:/arrayType
		s1 := dtypesym(t.Elem())
		t2 := types.NewSlice(t.Elem())
		s2 := dtypesym(t2)
		ot = dcommontype(lsym, ot, t)
		ot = dsymptr(lsym, ot, s1.Linksym(), 0)
		ot = dsymptr(lsym, ot, s2.Linksym(), 0)
		ot = duintptr(lsym, ot, uint64(t.NumElem()))
		ot = dextratype(lsym, ot, t, 0)

	case TSLICE:
		// ../../../../runtime/type.go:/sliceType
		s1 := dtypesym(t.Elem())
		ot = dcommontype(lsym, ot, t)
		ot = dsymptr(lsym, ot, s1.Linksym(), 0)
		ot = dextratype(lsym, ot, t, 0)

	case TCHAN:
		// ../../../../runtime/type.go:/chanType
		s1 := dtypesym(t.Elem())
		ot = dcommontype(lsym, ot, t)
		ot = dsymptr(lsym, ot, s1.Linksym(), 0)
		ot = duintptr(lsym, ot, uint64(t.ChanDir()))
		ot = dextratype(lsym, ot, t, 0)

	case TFUNC:
		for _, t1 := range t.Recvs().Fields().Slice() {
			dtypesym(t1.Type)
		}
		isddd := false
		for _, t1 := range t.Params().Fields().Slice() {
			isddd = t1.Isddd()
			dtypesym(t1.Type)
		}
		for _, t1 := range t.Results().Fields().Slice() {
			dtypesym(t1.Type)
		}

		ot = dcommontype(lsym, ot, t)
		inCount := t.Recvs().NumFields() + t.Params().NumFields()
		outCount := t.Results().NumFields()
		if isddd {
			outCount |= 1 << 15
		}
		ot = duint16(lsym, ot, uint16(inCount))
		ot = duint16(lsym, ot, uint16(outCount))
		if Widthptr == 8 {
			ot += 4 // align for *rtype
		}

		dataAdd := (inCount + t.Results().NumFields()) * Widthptr
		ot = dextratype(lsym, ot, t, dataAdd)

		// Array of rtype pointers follows funcType.
		for _, t1 := range t.Recvs().Fields().Slice() {
			ot = dsymptr(lsym, ot, dtypesym(t1.Type).Linksym(), 0)
		}
		for _, t1 := range t.Params().Fields().Slice() {
			ot = dsymptr(lsym, ot, dtypesym(t1.Type).Linksym(), 0)
		}
		for _, t1 := range t.Results().Fields().Slice() {
			ot = dsymptr(lsym, ot, dtypesym(t1.Type).Linksym(), 0)
		}

	case TINTER:
		m := imethods(t)
		n := len(m)
		for _, a := range m {
			dtypesym(a.type_)
		}

		// ../../../../runtime/type.go:/interfaceType
		ot = dcommontype(lsym, ot, t)

		var tpkg *types.Pkg
		if t.Sym != nil && t != types.Types[t.Etype] && t != types.Errortype {
			tpkg = t.Sym.Pkg
		}
		ot = dgopkgpath(lsym, ot, tpkg)

		ot = dsymptr(lsym, ot, lsym, ot+3*Widthptr+uncommonSize(t))
		ot = duintptr(lsym, ot, uint64(n))
		ot = duintptr(lsym, ot, uint64(n))
		dataAdd := imethodSize() * n
		ot = dextratype(lsym, ot, t, dataAdd)

		for _, a := range m {
			// ../../../../runtime/type.go:/imethod
			exported := exportname(a.name)
			var pkg *types.Pkg
			if !exported && a.pkg != tpkg {
				pkg = a.pkg
			}
			nsym := dname(a.name, "", pkg, exported)

			ot = dsymptrOff(lsym, ot, nsym, 0)
			ot = dsymptrOff(lsym, ot, dtypesym(a.type_).Linksym(), 0)
		}

	// ../../../../runtime/type.go:/mapType
	case TMAP:
		s1 := dtypesym(t.Key())
		s2 := dtypesym(t.Val())
		s3 := dtypesym(mapbucket(t))
		s4 := dtypesym(hmap(t))
		ot = dcommontype(lsym, ot, t)
		ot = dsymptr(lsym, ot, s1.Linksym(), 0)
		ot = dsymptr(lsym, ot, s2.Linksym(), 0)
		ot = dsymptr(lsym, ot, s3.Linksym(), 0)
		ot = dsymptr(lsym, ot, s4.Linksym(), 0)
		if t.Key().Width > MAXKEYSIZE {
			ot = duint8(lsym, ot, uint8(Widthptr))
			ot = duint8(lsym, ot, 1) // indirect
		} else {
			ot = duint8(lsym, ot, uint8(t.Key().Width))
			ot = duint8(lsym, ot, 0) // not indirect
		}

		if t.Val().Width > MAXVALSIZE {
			ot = duint8(lsym, ot, uint8(Widthptr))
			ot = duint8(lsym, ot, 1) // indirect
		} else {
			ot = duint8(lsym, ot, uint8(t.Val().Width))
			ot = duint8(lsym, ot, 0) // not indirect
		}

		ot = duint16(lsym, ot, uint16(mapbucket(t).Width))
		ot = duint8(lsym, ot, uint8(obj.Bool2int(isreflexive(t.Key()))))
		ot = duint8(lsym, ot, uint8(obj.Bool2int(needkeyupdate(t.Key()))))
		ot = dextratype(lsym, ot, t, 0)

	case TPTR32, TPTR64:
		if t.Elem().Etype == TANY {
			// ../../../../runtime/type.go:/UnsafePointerType
			ot = dcommontype(lsym, ot, t)
			ot = dextratype(lsym, ot, t, 0)

			break
		}

		// ../../../../runtime/type.go:/ptrType
		s1 := dtypesym(t.Elem())

		ot = dcommontype(lsym, ot, t)
		ot = dsymptr(lsym, ot, s1.Linksym(), 0)
		ot = dextratype(lsym, ot, t, 0)

	// ../../../../runtime/type.go:/structType
	// for security, only the exported fields.
	case TSTRUCT:
		n := 0

		for _, t1 := range t.Fields().Slice() {
			dtypesym(t1.Type)
			n++
		}

		ot = dcommontype(lsym, ot, t)
		pkg := localpkg
		if t.Sym != nil {
			pkg = t.Sym.Pkg
		} else {
			// Unnamed type. Grab the package from the first field, if any.
			for _, f := range t.Fields().Slice() {
				if f.Embedded != 0 {
					continue
				}
				pkg = f.Sym.Pkg
				break
			}
		}
		ot = dgopkgpath(lsym, ot, pkg)
		ot = dsymptr(lsym, ot, lsym, ot+3*Widthptr+uncommonSize(t))
		ot = duintptr(lsym, ot, uint64(n))
		ot = duintptr(lsym, ot, uint64(n))

		dataAdd := n * structfieldSize()
		ot = dextratype(lsym, ot, t, dataAdd)

		for _, f := range t.Fields().Slice() {
			// ../../../../runtime/type.go:/structField
			ot = dnameField(lsym, ot, pkg, f)
			ot = dsymptr(lsym, ot, dtypesym(f.Type).Linksym(), 0)
			offsetAnon := uint64(f.Offset) << 1
			if offsetAnon>>1 != uint64(f.Offset) {
				Fatalf("%v: bad field offset for %s", t, f.Sym.Name)
			}
			if f.Embedded != 0 {
				offsetAnon |= 1
			}
			ot = duintptr(lsym, ot, offsetAnon)
		}
	}

	ot = dextratypeData(lsym, ot, t)
	ggloblsym(lsym, int32(ot), int16(dupok|obj.RODATA))

	// The linker will leave a table of all the typelinks for
	// types in the binary, so the runtime can find them.
	//
	// When buildmode=shared, all types are in typelinks so the
	// runtime can deduplicate type pointers.
	keep := Ctxt.Flag_dynlink
	if !keep && t.Sym == nil {
		// For an unnamed type, we only need the link if the type can
		// be created at run time by reflect.PtrTo and similar
		// functions. If the type exists in the program, those
		// functions must return the existing type structure rather
		// than creating a new one.
		switch t.Etype {
		case TPTR32, TPTR64, TARRAY, TCHAN, TFUNC, TMAP, TSLICE, TSTRUCT:
			keep = true
		}
	}
	lsym.Set(obj.AttrMakeTypelink, keep)

	return s
}

// for each itabEntry, gather the methods on
// the concrete type that implement the interface
func peekitabs() {
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
	if len(sigs) == 0 {
		return nil
	}

	// both sigs and methods are sorted by name,
	// so we can find the intersect in a single pass
	for _, m := range methods {
		if m.name == sigs[0].name {
			out = append(out, m.isym.Linksym())
			sigs = sigs[1:]
			if len(sigs) == 0 {
				break
			}
		}
	}

	return out
}

// itabsym uses the information gathered in
// peekitabs to de-virtualize interface methods.
// Since this is called by the SSA backend, it shouldn't
// generate additional Nodes, Syms, etc.
func itabsym(it *obj.LSym, offset int64) *obj.LSym {
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
	methodnum := int((offset - 3*int64(Widthptr) - 8) / int64(Widthptr))
	if methodnum >= len(syms) {
		return nil
	}
	return syms[methodnum]
}

func addsignat(t *types.Type) {
	signatset[t] = struct{}{}
}

func addsignats(dcls []*Node) {
	// copy types from dcl list to signatset
	for _, n := range dcls {
		if n.Op == OTYPE {
			addsignat(n.Type)
		}
	}
}

func dumpsignats() {
	// Process signatset. Use a loop, as dtypesym adds
	// entries to signatset while it is being processed.
	signats := make([]typeAndStr, len(signatset))
	for len(signatset) > 0 {
		signats = signats[:0]
		// Transfer entries to a slice and sort, for reproducible builds.
		for t := range signatset {
			signats = append(signats, typeAndStr{t: t, short: typesymname(t), regular: t.String()})
			delete(signatset, t)
		}
		sort.Sort(typesByString(signats))
		for _, ts := range signats {
			t := ts.t
			dtypesym(t)
			if t.Sym != nil {
				dtypesym(types.NewPtr(t))
			}
		}
	}
}

func dumptabs() {
	// process itabs
	for _, i := range itabs {
		// dump empty itab symbol into i.sym
		// type itab struct {
		//   inter  *interfacetype
		//   _type  *_type
		//   link   *itab
		//   hash   uint32
		//   bad    bool
		//   inhash bool
		//   unused [2]byte
		//   fun    [1]uintptr // variable sized
		// }
		o := dsymptr(i.lsym, 0, dtypesym(i.itype).Linksym(), 0)
		o = dsymptr(i.lsym, o, dtypesym(i.t).Linksym(), 0)
		o += Widthptr                          // skip link field
		o = duint32(i.lsym, o, typehash(i.t))  // copy of type hash
		o += 4                                 // skip bad/inhash/unused fields
		o += len(imethods(i.itype)) * Widthptr // skip fun method pointers
		// at runtime the itab will contain pointers to types, other itabs and
		// method functions. None are allocated on heap, so we can use obj.NOPTR.
		ggloblsym(i.lsym, int32(o), int16(obj.DUPOK|obj.NOPTR))

		ilink := itablinkpkg.Lookup(i.t.ShortString() + "," + i.itype.ShortString()).Linksym()
		dsymptr(ilink, 0, i.lsym, 0)
		ggloblsym(ilink, int32(Widthptr), int16(obj.DUPOK|obj.RODATA))
	}

	// process ptabs
	if localpkg.Name == "main" && len(ptabs) > 0 {
		ot := 0
		s := Ctxt.Lookup("go.plugin.tabs")
		for _, p := range ptabs {
			// Dump ptab symbol into go.pluginsym package.
			//
			// type ptab struct {
			//	name nameOff
			//	typ  typeOff // pointer to symbol
			// }
			nsym := dname(p.s.Name, "", nil, true)
			ot = dsymptrOff(s, ot, nsym, 0)
			ot = dsymptrOff(s, ot, dtypesym(p.t).Linksym(), 0)
		}
		ggloblsym(s, int32(ot), int16(obj.RODATA))

		ot = 0
		s = Ctxt.Lookup("go.plugin.exports")
		for _, p := range ptabs {
			ot = dsymptr(s, ot, p.s.Linksym(), 0)
		}
		ggloblsym(s, int32(ot), int16(obj.RODATA))
	}
}

func dumpimportstrings() {
	// generate import strings for imported packages
	for _, p := range types.ImportedPkgList() {
		dimportpath(p)
	}
}

func dumpbasictypes() {
	// do basic types if compiling package runtime.
	// they have to be in at least one package,
	// and runtime is always loaded implicitly,
	// so this is as good as any.
	// another possible choice would be package main,
	// but using runtime means fewer copies in object files.
	if myimportpath == "runtime" {
		for i := types.EType(1); i <= TBOOL; i++ {
			dtypesym(types.NewPtr(types.Types[i]))
		}
		dtypesym(types.NewPtr(types.Types[TSTRING]))
		dtypesym(types.NewPtr(types.Types[TUNSAFEPTR]))

		// emit type structs for error and func(error) string.
		// The latter is the type of an auto-generated wrapper.
		dtypesym(types.NewPtr(types.Errortype))

		dtypesym(functype(nil, []*Node{anonfield(types.Errortype)}, []*Node{anonfield(types.Types[TSTRING])}))

		// add paths for runtime and main, which 6l imports implicitly.
		dimportpath(Runtimepkg)

		if flag_race {
			dimportpath(racepkg)
		}
		if flag_msan {
			dimportpath(msanpkg)
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
	return a[i].regular < a[j].regular
}
func (a typesByString) Swap(i, j int) { a[i], a[j] = a[j], a[i] }

func dalgsym(t *types.Type) *obj.LSym {
	var lsym *obj.LSym
	var hashfunc *obj.LSym
	var eqfunc *obj.LSym

	// dalgsym is only called for a type that needs an algorithm table,
	// which implies that the type is comparable (or else it would use ANOEQ).

	if algtype(t) == AMEM {
		// we use one algorithm table for all AMEM types of a given size
		p := fmt.Sprintf(".alg%d", t.Width)

		s := typeLookup(p)
		lsym = s.Linksym()
		if s.AlgGen() {
			return lsym
		}
		s.SetAlgGen(true)

		if memhashvarlen == nil {
			memhashvarlen = Sysfunc("memhash_varlen")
			memequalvarlen = Sysfunc("memequal_varlen")
		}

		// make hash closure
		p = fmt.Sprintf(".hashfunc%d", t.Width)

		hashfunc = typeLookup(p).Linksym()

		ot := 0
		ot = dsymptr(hashfunc, ot, memhashvarlen, 0)
		ot = duintptr(hashfunc, ot, uint64(t.Width)) // size encoded in closure
		ggloblsym(hashfunc, int32(ot), obj.DUPOK|obj.RODATA)

		// make equality closure
		p = fmt.Sprintf(".eqfunc%d", t.Width)

		eqfunc = typeLookup(p).Linksym()

		ot = 0
		ot = dsymptr(eqfunc, ot, memequalvarlen, 0)
		ot = duintptr(eqfunc, ot, uint64(t.Width))
		ggloblsym(eqfunc, int32(ot), obj.DUPOK|obj.RODATA)
	} else {
		// generate an alg table specific to this type
		s := typesymprefix(".alg", t)
		lsym = s.Linksym()

		hash := typesymprefix(".hash", t)
		eq := typesymprefix(".eq", t)
		hashfunc = typesymprefix(".hashfunc", t).Linksym()
		eqfunc = typesymprefix(".eqfunc", t).Linksym()

		genhash(hash, t)
		geneq(eq, t)

		// make Go funcs (closures) for calling hash and equal from Go
		dsymptr(hashfunc, 0, hash.Linksym(), 0)
		ggloblsym(hashfunc, int32(Widthptr), obj.DUPOK|obj.RODATA)
		dsymptr(eqfunc, 0, eq.Linksym(), 0)
		ggloblsym(eqfunc, int32(Widthptr), obj.DUPOK|obj.RODATA)
	}

	// ../../../../runtime/alg.go:/typeAlg
	ot := 0

	ot = dsymptr(lsym, ot, hashfunc, 0)
	ot = dsymptr(lsym, ot, eqfunc, 0)
	ggloblsym(lsym, int32(ot), obj.DUPOK|obj.RODATA)
	return lsym
}

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
	ptrdata = typeptrdata(t)
	if ptrdata/int64(Widthptr) <= maxPtrmaskBytes*8 {
		lsym = dgcptrmask(t)
		return
	}

	useGCProg = true
	lsym, ptrdata = dgcprog(t)
	return
}

// dgcptrmask emits and returns the symbol containing a pointer mask for type t.
func dgcptrmask(t *types.Type) *obj.LSym {
	ptrmask := make([]byte, (typeptrdata(t)/int64(Widthptr)+7)/8)
	fillptrmask(t, ptrmask)
	p := fmt.Sprintf("gcbits.%x", ptrmask)

	sym := Runtimepkg.Lookup(p)
	lsym := sym.Linksym()
	if !sym.Uniq() {
		sym.SetUniq(true)
		for i, x := range ptrmask {
			duint8(lsym, i, x)
		}
		ggloblsym(lsym, int32(len(ptrmask)), obj.DUPOK|obj.RODATA|obj.LOCAL)
	}
	return lsym
}

// fillptrmask fills in ptrmask with 1s corresponding to the
// word offsets in t that hold pointers.
// ptrmask is assumed to fit at least typeptrdata(t)/Widthptr bits.
func fillptrmask(t *types.Type, ptrmask []byte) {
	for i := range ptrmask {
		ptrmask[i] = 0
	}
	if !types.Haspointers(t) {
		return
	}

	vec := bvalloc(8 * int32(len(ptrmask)))
	xoffset := int64(0)
	onebitwalktype1(t, &xoffset, vec)

	nptr := typeptrdata(t) / int64(Widthptr)
	for i := int64(0); i < nptr; i++ {
		if vec.Get(int32(i)) {
			ptrmask[i/8] |= 1 << (uint(i) % 8)
		}
	}
}

// dgcprog emits and returns the symbol containing a GC program for type t
// along with the size of the data described by the program (in the range [typeptrdata(t), t.Width]).
// In practice, the size is typeptrdata(t) except for non-trivial arrays.
// For non-trivial arrays, the program describes the full t.Width size.
func dgcprog(t *types.Type) (*obj.LSym, int64) {
	dowidth(t)
	if t.Width == BADWIDTH {
		Fatalf("dgcprog: %v badwidth", t)
	}
	lsym := typesymprefix(".gcprog", t).Linksym()
	var p GCProg
	p.init(lsym)
	p.emit(t, 0)
	offset := p.w.BitIndex() * int64(Widthptr)
	p.end()
	if ptrdata := typeptrdata(t); offset < ptrdata || offset > t.Width {
		Fatalf("dgcprog: %v: offset=%d but ptrdata=%d size=%d", t, offset, ptrdata, t.Width)
	}
	return lsym, offset
}

type GCProg struct {
	lsym   *obj.LSym
	symoff int
	w      gcprog.Writer
}

var Debug_gcprog int // set by -d gcprog

func (p *GCProg) init(lsym *obj.LSym) {
	p.lsym = lsym
	p.symoff = 4 // first 4 bytes hold program length
	p.w.Init(p.writeByte)
	if Debug_gcprog > 0 {
		fmt.Fprintf(os.Stderr, "compile: start GCProg for %v\n", lsym)
		p.w.Debug(os.Stderr)
	}
}

func (p *GCProg) writeByte(x byte) {
	p.symoff = duint8(p.lsym, p.symoff, x)
}

func (p *GCProg) end() {
	p.w.End()
	duint32(p.lsym, 0, uint32(p.symoff-4))
	ggloblsym(p.lsym, int32(p.symoff), obj.DUPOK|obj.RODATA|obj.LOCAL)
	if Debug_gcprog > 0 {
		fmt.Fprintf(os.Stderr, "compile: end GCProg for %v\n", p.lsym)
	}
}

func (p *GCProg) emit(t *types.Type, offset int64) {
	dowidth(t)
	if !types.Haspointers(t) {
		return
	}
	if t.Width == int64(Widthptr) {
		p.w.Ptr(offset / int64(Widthptr))
		return
	}
	switch t.Etype {
	default:
		Fatalf("GCProg.emit: unexpected type %v", t)

	case TSTRING:
		p.w.Ptr(offset / int64(Widthptr))

	case TINTER:
		p.w.Ptr(offset / int64(Widthptr))
		p.w.Ptr(offset/int64(Widthptr) + 1)

	case TSLICE:
		p.w.Ptr(offset / int64(Widthptr))

	case TARRAY:
		if t.NumElem() == 0 {
			// should have been handled by haspointers check above
			Fatalf("GCProg.emit: empty array")
		}

		// Flatten array-of-array-of-array to just a big array by multiplying counts.
		count := t.NumElem()
		elem := t.Elem()
		for elem.IsArray() {
			count *= elem.NumElem()
			elem = elem.Elem()
		}

		if !p.w.ShouldRepeat(elem.Width/int64(Widthptr), count) {
			// Cheaper to just emit the bits.
			for i := int64(0); i < count; i++ {
				p.emit(elem, offset+i*elem.Width)
			}
			return
		}
		p.emit(elem, offset)
		p.w.ZeroUntil((offset + elem.Width) / int64(Widthptr))
		p.w.Repeat(elem.Width/int64(Widthptr), count-1)

	case TSTRUCT:
		for _, t1 := range t.Fields().Slice() {
			p.emit(t1.Type, offset+t1.Offset)
		}
	}
}

// zeroaddr returns the address of a symbol with at least
// size bytes of zeros.
func zeroaddr(size int64) *Node {
	if size >= 1<<31 {
		Fatalf("map value too big %d", size)
	}
	if zerosize < size {
		zerosize = size
	}
	s := mappkg.Lookup("zero")
	if s.Def == nil {
		x := newname(s)
		x.Type = types.Types[TUINT8]
		x.SetClass(PEXTERN)
		x.SetTypecheck(1)
		s.Def = asTypesNode(x)
	}
	z := nod(OADDR, asNode(s.Def), nil)
	z.Type = types.NewPtr(types.Types[TUINT8])
	z.SetAddable(true)
	z.SetTypecheck(1)
	return z
}

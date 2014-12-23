// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/internal/obj"
	"fmt"
)

/*
 * runtime interface and reflection data structures
 */
var signatlist *NodeList

func sigcmp(a *Sig, b *Sig) int {
	i := stringsCompare(a.name, b.name)
	if i != 0 {
		return i
	}
	if a.pkg == b.pkg {
		return 0
	}
	if a.pkg == nil {
		return -1
	}
	if b.pkg == nil {
		return +1
	}
	return stringsCompare(a.pkg.Path, b.pkg.Path)
}

func lsort(l *Sig, f func(*Sig, *Sig) int) *Sig {
	if l == nil || l.link == nil {
		return l
	}

	l1 := l
	l2 := l
	for {
		l2 = l2.link
		if l2 == nil {
			break
		}
		l2 = l2.link
		if l2 == nil {
			break
		}
		l1 = l1.link
	}

	l2 = l1.link
	l1.link = nil
	l1 = lsort(l, f)
	l2 = lsort(l2, f)

	/* set up lead element */
	if f(l1, l2) < 0 {
		l = l1
		l1 = l1.link
	} else {
		l = l2
		l2 = l2.link
	}

	le := l

	for {
		if l1 == nil {
			for l2 != nil {
				le.link = l2
				le = l2
				l2 = l2.link
			}

			le.link = nil
			break
		}

		if l2 == nil {
			for l1 != nil {
				le.link = l1
				le = l1
				l1 = l1.link
			}

			break
		}

		if f(l1, l2) < 0 {
			le.link = l1
			le = l1
			l1 = l1.link
		} else {
			le.link = l2
			le = l2
			l2 = l2.link
		}
	}

	le.link = nil
	return l
}

// Builds a type respresenting a Bucket structure for
// the given map type.  This type is not visible to users -
// we include only enough information to generate a correct GC
// program for it.
// Make sure this stays in sync with ../../runtime/hashmap.go!
const (
	BUCKETSIZE = 8
	MAXKEYSIZE = 128
	MAXVALSIZE = 128
)

func makefield(name string, t *Type) *Type {
	f := typ(TFIELD)
	f.Type = t
	f.Sym = new(Sym)
	f.Sym.Name = name
	return f
}

func mapbucket(t *Type) *Type {
	if t.Bucket != nil {
		return t.Bucket
	}

	bucket := typ(TSTRUCT)
	keytype := t.Down
	valtype := t.Type
	dowidth(keytype)
	dowidth(valtype)
	if keytype.Width > MAXKEYSIZE {
		keytype = Ptrto(keytype)
	}
	if valtype.Width > MAXVALSIZE {
		valtype = Ptrto(valtype)
	}

	// The first field is: uint8 topbits[BUCKETSIZE].
	arr := typ(TARRAY)

	arr.Type = Types[TUINT8]
	arr.Bound = BUCKETSIZE
	var field [4]*Type
	field[0] = makefield("topbits", arr)
	arr = typ(TARRAY)
	arr.Type = keytype
	arr.Bound = BUCKETSIZE
	field[1] = makefield("keys", arr)
	arr = typ(TARRAY)
	arr.Type = valtype
	arr.Bound = BUCKETSIZE
	field[2] = makefield("values", arr)
	field[3] = makefield("overflow", Ptrto(bucket))

	// link up fields
	bucket.Noalg = 1

	bucket.Local = t.Local
	bucket.Type = field[0]
	for n := int32(0); n < int32(len(field)-1); n++ {
		field[n].Down = field[n+1]
	}
	field[len(field)-1].Down = nil
	dowidth(bucket)

	// Pad to the native integer alignment.
	// This is usually the same as widthptr; the exception (as usual) is amd64p32.
	if Widthreg > Widthptr {
		bucket.Width += int64(Widthreg) - int64(Widthptr)
	}

	// See comment on hmap.overflow in ../../runtime/hashmap.go.
	if !haspointers(t.Type) && !haspointers(t.Down) && t.Type.Width <= MAXKEYSIZE && t.Down.Width <= MAXVALSIZE {
		bucket.Haspointers = 1 // no pointers
	}

	t.Bucket = bucket

	bucket.Map = t
	return bucket
}

// Builds a type representing a Hmap structure for the given map type.
// Make sure this stays in sync with ../../runtime/hashmap.go!
func hmap(t *Type) *Type {
	if t.Hmap != nil {
		return t.Hmap
	}

	bucket := mapbucket(t)
	var field [8]*Type
	field[0] = makefield("count", Types[TINT])
	field[1] = makefield("flags", Types[TUINT8])
	field[2] = makefield("B", Types[TUINT8])
	field[3] = makefield("hash0", Types[TUINT32])
	field[4] = makefield("buckets", Ptrto(bucket))
	field[5] = makefield("oldbuckets", Ptrto(bucket))
	field[6] = makefield("nevacuate", Types[TUINTPTR])
	field[7] = makefield("overflow", Types[TUNSAFEPTR])

	h := typ(TSTRUCT)
	h.Noalg = 1
	h.Local = t.Local
	h.Type = field[0]
	for n := int32(0); n < int32(len(field)-1); n++ {
		field[n].Down = field[n+1]
	}
	field[len(field)-1].Down = nil
	dowidth(h)
	t.Hmap = h
	h.Map = t
	return h
}

func hiter(t *Type) *Type {
	if t.Hiter != nil {
		return t.Hiter
	}

	// build a struct:
	// hash_iter {
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
	// must match ../../runtime/hashmap.go:hash_iter.
	var field [12]*Type
	field[0] = makefield("key", Ptrto(t.Down))

	field[1] = makefield("val", Ptrto(t.Type))
	field[2] = makefield("t", Ptrto(Types[TUINT8]))
	field[3] = makefield("h", Ptrto(hmap(t)))
	field[4] = makefield("buckets", Ptrto(mapbucket(t)))
	field[5] = makefield("bptr", Ptrto(mapbucket(t)))
	field[6] = makefield("overflow0", Types[TUNSAFEPTR])
	field[7] = makefield("overflow1", Types[TUNSAFEPTR])
	field[8] = makefield("startBucket", Types[TUINTPTR])
	field[9] = makefield("stuff", Types[TUINTPTR]) // offset+wrapped+B+I
	field[10] = makefield("bucket", Types[TUINTPTR])
	field[11] = makefield("checkBucket", Types[TUINTPTR])

	// build iterator struct holding the above fields
	i := typ(TSTRUCT)

	i.Noalg = 1
	i.Type = field[0]
	for n := int32(0); n < int32(len(field)-1); n++ {
		field[n].Down = field[n+1]
	}
	field[len(field)-1].Down = nil
	dowidth(i)
	if i.Width != int64(12*Widthptr) {
		Yyerror("hash_iter size not correct %d %d", i.Width, 12*Widthptr)
	}
	t.Hiter = i
	i.Map = t
	return i
}

/*
 * f is method type, with receiver.
 * return function type, receiver as first argument (or not).
 */
func methodfunc(f *Type, receiver *Type) *Type {
	var in *NodeList
	if receiver != nil {
		d := Nod(ODCLFIELD, nil, nil)
		d.Type = receiver
		in = list(in, d)
	}

	var d *Node
	for t := getinargx(f).Type; t != nil; t = t.Down {
		d = Nod(ODCLFIELD, nil, nil)
		d.Type = t.Type
		d.Isddd = t.Isddd
		in = list(in, d)
	}

	var out *NodeList
	for t := getoutargx(f).Type; t != nil; t = t.Down {
		d = Nod(ODCLFIELD, nil, nil)
		d.Type = t.Type
		out = list(out, d)
	}

	t := functype(nil, in, out)
	if f.Nname != nil {
		// Link to name of original method function.
		t.Nname = f.Nname
	}

	return t
}

/*
 * return methods of non-interface type t, sorted by name.
 * generates stub functions as needed.
 */
func methods(t *Type) *Sig {
	// method type
	mt := methtype(t, 0)

	if mt == nil {
		return nil
	}
	expandmeth(mt)

	// type stored in interface word
	it := t

	if !isdirectiface(it) {
		it = Ptrto(t)
	}

	// make list of methods for t,
	// generating code if necessary.
	var a *Sig

	var this *Type
	var b *Sig
	var method *Sym
	for f := mt.Xmethod; f != nil; f = f.Down {
		if f.Etype != TFIELD {
			Fatal("methods: not field %v", Tconv(f, 0))
		}
		if f.Type.Etype != TFUNC || f.Type.Thistuple == 0 {
			Fatal("non-method on %v method %v %v\n", Tconv(mt, 0), Sconv(f.Sym, 0), Tconv(f, 0))
		}
		if getthisx(f.Type).Type == nil {
			Fatal("receiver with no type on %v method %v %v\n", Tconv(mt, 0), Sconv(f.Sym, 0), Tconv(f, 0))
		}
		if f.Nointerface {
			continue
		}

		method = f.Sym
		if method == nil {
			continue
		}

		// get receiver type for this particular method.
		// if pointer receiver but non-pointer t and
		// this is not an embedded pointer inside a struct,
		// method does not apply.
		this = getthisx(f.Type).Type.Type

		if Isptr[this.Etype] && this.Type == t {
			continue
		}
		if Isptr[this.Etype] && !Isptr[t.Etype] && f.Embedded != 2 && !isifacemethod(f.Type) {
			continue
		}

		b = new(Sig)
		b.link = a
		a = b

		a.name = method.Name
		if !exportname(method.Name) {
			if method.Pkg == nil {
				Fatal("methods: missing package")
			}
			a.pkg = method.Pkg
		}

		a.isym = methodsym(method, it, 1)
		a.tsym = methodsym(method, t, 0)
		a.type_ = methodfunc(f.Type, t)
		a.mtype = methodfunc(f.Type, nil)

		if a.isym.Flags&SymSiggen == 0 {
			a.isym.Flags |= SymSiggen
			if !Eqtype(this, it) || this.Width < Types[Tptr].Width {
				compiling_wrappers = 1
				genwrapper(it, f, a.isym, 1)
				compiling_wrappers = 0
			}
		}

		if a.tsym.Flags&SymSiggen == 0 {
			a.tsym.Flags |= SymSiggen
			if !Eqtype(this, t) {
				compiling_wrappers = 1
				genwrapper(t, f, a.tsym, 0)
				compiling_wrappers = 0
			}
		}
	}

	return lsort(a, sigcmp)
}

/*
 * return methods of interface type t, sorted by name.
 */
func imethods(t *Type) *Sig {
	var a *Sig
	var method *Sym
	var isym *Sym

	var all *Sig
	var last *Sig
	for f := t.Type; f != nil; f = f.Down {
		if f.Etype != TFIELD {
			Fatal("imethods: not field")
		}
		if f.Type.Etype != TFUNC || f.Sym == nil {
			continue
		}
		method = f.Sym
		a = new(Sig)
		a.name = method.Name
		if !exportname(method.Name) {
			if method.Pkg == nil {
				Fatal("imethods: missing package")
			}
			a.pkg = method.Pkg
		}

		a.mtype = f.Type
		a.offset = 0
		a.type_ = methodfunc(f.Type, nil)

		if last != nil && sigcmp(last, a) >= 0 {
			Fatal("sigcmp vs sortinter %s %s", last.name, a.name)
		}
		if last == nil {
			all = a
		} else {
			last.link = a
		}
		last = a

		// Compiler can only refer to wrappers for non-blank methods.
		if isblanksym(method) {
			continue
		}

		// NOTE(rsc): Perhaps an oversight that
		// IfaceType.Method is not in the reflect data.
		// Generate the method body, so that compiled
		// code can refer to it.
		isym = methodsym(method, t, 0)

		if isym.Flags&SymSiggen == 0 {
			isym.Flags |= SymSiggen
			genwrapper(t, f, isym, 0)
		}
	}

	return all
}

var dimportpath_gopkg *Pkg

func dimportpath(p *Pkg) {
	if p.Pathsym != nil {
		return
	}

	// If we are compiling the runtime package, there are two runtime packages around
	// -- localpkg and Runtimepkg.  We don't want to produce import path symbols for
	// both of them, so just produce one for localpkg.
	if myimportpath == "runtime" && p == Runtimepkg {
		return
	}

	if dimportpath_gopkg == nil {
		dimportpath_gopkg = mkpkg("go")
		dimportpath_gopkg.Name = "go"
	}

	var nam string
	if p == localpkg {
		// Note: myimportpath != "", or else dgopkgpath won't call dimportpath.
		nam = "importpath." + pathtoprefix(myimportpath) + "."
	} else {
		nam = "importpath." + p.Prefix + "."
	}

	n := Nod(ONAME, nil, nil)
	n.Sym = Pkglookup(nam, dimportpath_gopkg)

	n.Class = PEXTERN
	n.Xoffset = 0
	p.Pathsym = n.Sym

	gdatastring(n, p.Path)
	ggloblsym(n.Sym, int32(Types[TSTRING].Width), obj.DUPOK|obj.RODATA)
}

func dgopkgpath(s *Sym, ot int, pkg *Pkg) int {
	if pkg == nil {
		return dgostringptr(s, ot, "")
	}

	if pkg == localpkg && myimportpath == "" {
		// If we don't know the full path of the package being compiled (i.e. -p
		// was not passed on the compiler command line), emit reference to
		// go.importpath.""., which 6l will rewrite using the correct import path.
		// Every package that imports this one directly defines the symbol.
		var ns *Sym

		if ns == nil {
			ns = Pkglookup("importpath.\"\".", mkpkg("go"))
		}
		return dsymptr(s, ot, ns, 0)
	}

	dimportpath(pkg)
	return dsymptr(s, ot, pkg.Pathsym, 0)
}

/*
 * uncommonType
 * ../../runtime/type.go:/uncommonType
 */
func dextratype(sym *Sym, off int, t *Type, ptroff int) int {
	m := methods(t)
	if t.Sym == nil && m == nil {
		return off
	}

	// fill in *extraType pointer in header
	off = int(Rnd(int64(off), int64(Widthptr)))

	dsymptr(sym, ptroff, sym, off)

	n := 0
	for a := m; a != nil; a = a.link {
		dtypesym(a.type_)
		n++
	}

	ot := off
	s := sym
	if t.Sym != nil {
		ot = dgostringptr(s, ot, t.Sym.Name)
		if t != Types[t.Etype] && t != errortype {
			ot = dgopkgpath(s, ot, t.Sym.Pkg)
		} else {
			ot = dgostringptr(s, ot, "")
		}
	} else {
		ot = dgostringptr(s, ot, "")
		ot = dgostringptr(s, ot, "")
	}

	// slice header
	ot = dsymptr(s, ot, s, ot+Widthptr+2*Widthint)

	ot = duintxx(s, ot, uint64(n), Widthint)
	ot = duintxx(s, ot, uint64(n), Widthint)

	// methods
	for a := m; a != nil; a = a.link {
		// method
		// ../../runtime/type.go:/method
		ot = dgostringptr(s, ot, a.name)

		ot = dgopkgpath(s, ot, a.pkg)
		ot = dsymptr(s, ot, dtypesym(a.mtype), 0)
		ot = dsymptr(s, ot, dtypesym(a.type_), 0)
		if a.isym != nil {
			ot = dsymptr(s, ot, a.isym, 0)
		} else {
			ot = duintptr(s, ot, 0)
		}
		if a.tsym != nil {
			ot = dsymptr(s, ot, a.tsym, 0)
		} else {
			ot = duintptr(s, ot, 0)
		}
	}

	return ot
}

var kinds = []int{
	TINT:        obj.KindInt,
	TUINT:       obj.KindUint,
	TINT8:       obj.KindInt8,
	TUINT8:      obj.KindUint8,
	TINT16:      obj.KindInt16,
	TUINT16:     obj.KindUint16,
	TINT32:      obj.KindInt32,
	TUINT32:     obj.KindUint32,
	TINT64:      obj.KindInt64,
	TUINT64:     obj.KindUint64,
	TUINTPTR:    obj.KindUintptr,
	TFLOAT32:    obj.KindFloat32,
	TFLOAT64:    obj.KindFloat64,
	TBOOL:       obj.KindBool,
	TSTRING:     obj.KindString,
	TPTR32:      obj.KindPtr,
	TPTR64:      obj.KindPtr,
	TSTRUCT:     obj.KindStruct,
	TINTER:      obj.KindInterface,
	TCHAN:       obj.KindChan,
	TMAP:        obj.KindMap,
	TARRAY:      obj.KindArray,
	TFUNC:       obj.KindFunc,
	TCOMPLEX64:  obj.KindComplex64,
	TCOMPLEX128: obj.KindComplex128,
	TUNSAFEPTR:  obj.KindUnsafePointer,
}

func haspointers(t *Type) bool {
	if t.Haspointers != 0 {
		return t.Haspointers-1 != 0
	}

	var ret bool
	switch t.Etype {
	case TINT,
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
		TFLOAT32,
		TFLOAT64,
		TCOMPLEX64,
		TCOMPLEX128,
		TBOOL:
		ret = false

	case TARRAY:
		if t.Bound < 0 { // slice
			ret = true
			break
		}

		if t.Bound == 0 { // empty array
			ret = false
			break
		}

		ret = haspointers(t.Type)

	case TSTRUCT:
		ret = false
		for t1 := t.Type; t1 != nil; t1 = t1.Down {
			if haspointers(t1.Type) {
				ret = true
				break
			}
		}

	case TSTRING,
		TPTR32,
		TPTR64,
		TUNSAFEPTR,
		TINTER,
		TCHAN,
		TMAP,
		TFUNC:
		fallthrough
	default:
		ret = true
	}

	t.Haspointers = 1 + uint8(bool2int(ret))
	return ret
}

/*
 * commonType
 * ../../runtime/type.go:/commonType
 */

var dcommontype_algarray *Sym

func dcommontype(s *Sym, ot int, t *Type) int {
	if ot != 0 {
		Fatal("dcommontype %d", ot)
	}

	sizeofAlg := 2 * Widthptr
	if dcommontype_algarray == nil {
		dcommontype_algarray = Pkglookup("algarray", Runtimepkg)
	}
	dowidth(t)
	alg := algtype(t)
	var algsym *Sym
	if alg < 0 || alg == AMEM {
		algsym = dalgsym(t)
	}

	var sptr *Sym
	if t.Sym != nil && !Isptr[t.Etype] {
		sptr = dtypesym(Ptrto(t))
	} else {
		sptr = weaktypesym(Ptrto(t))
	}

	// All (non-reflect-allocated) Types share the same zero object.
	// Each place in the compiler where a pointer to the zero object
	// might be returned by a runtime call (map access return value,
	// 2-arg type cast) declares the size of the zerovalue it needs.
	// The linker magically takes the max of all the sizes.
	zero := Pkglookup("zerovalue", Runtimepkg)

	// We use size 0 here so we get the pointer to the zero value,
	// but don't allocate space for the zero value unless we need it.
	// TODO: how do we get this symbol into bss?  We really want
	// a read-only bss, but I don't think such a thing exists.

	// ../../pkg/reflect/type.go:/^type.commonType
	// actual type structure
	//	type commonType struct {
	//		size          uintptr
	//		hash          uint32
	//		_             uint8
	//		align         uint8
	//		fieldAlign    uint8
	//		kind          uint8
	//		alg           unsafe.Pointer
	//		gc            unsafe.Pointer
	//		string        *string
	//		*extraType
	//		ptrToThis     *Type
	//		zero          unsafe.Pointer
	//	}
	ot = duintptr(s, ot, uint64(t.Width))

	ot = duint32(s, ot, typehash(t))
	ot = duint8(s, ot, 0) // unused

	// runtime (and common sense) expects alignment to be a power of two.
	i := int(t.Align)

	if i == 0 {
		i = 1
	}
	if i&(i-1) != 0 {
		Fatal("invalid alignment %d for %v", t.Align, Tconv(t, 0))
	}
	ot = duint8(s, ot, t.Align) // align
	ot = duint8(s, ot, t.Align) // fieldAlign

	gcprog := usegcprog(t)

	i = kinds[t.Etype]
	if t.Etype == TARRAY && t.Bound < 0 {
		i = obj.KindSlice
	}
	if !haspointers(t) {
		i |= obj.KindNoPointers
	}
	if isdirectiface(t) {
		i |= obj.KindDirectIface
	}
	if gcprog {
		i |= obj.KindGCProg
	}
	ot = duint8(s, ot, uint8(i)) // kind
	if algsym == nil {
		ot = dsymptr(s, ot, dcommontype_algarray, alg*sizeofAlg)
	} else {
		ot = dsymptr(s, ot, algsym, 0)
	}

	// gc
	if gcprog {
		var gcprog1 *Sym
		var gcprog0 *Sym
		gengcprog(t, &gcprog0, &gcprog1)
		if gcprog0 != nil {
			ot = dsymptr(s, ot, gcprog0, 0)
		} else {
			ot = duintptr(s, ot, 0)
		}
		ot = dsymptr(s, ot, gcprog1, 0)
	} else {
		var gcmask [16]uint8
		gengcmask(t, gcmask[:])
		x1 := uint64(0)
		for i := 0; i < 8; i++ {
			x1 = x1<<8 | uint64(gcmask[i])
		}
		var p string
		if Widthptr == 4 {
			p = fmt.Sprintf("gcbits.0x%016x", x1)
		} else {
			x2 := uint64(0)
			for i := 0; i < 8; i++ {
				x2 = x2<<8 | uint64(gcmask[i+8])
			}
			p = fmt.Sprintf("gcbits.0x%016x%016x", x1, x2)
		}

		sbits := Pkglookup(p, Runtimepkg)
		if sbits.Flags&SymUniq == 0 {
			sbits.Flags |= SymUniq
			for i := 0; i < 2*Widthptr; i++ {
				duint8(sbits, i, gcmask[i])
			}
			ggloblsym(sbits, 2*int32(Widthptr), obj.DUPOK|obj.RODATA)
		}

		ot = dsymptr(s, ot, sbits, 0)
		ot = duintptr(s, ot, 0)
	}

	p := Tconv(t, obj.FmtLeft|obj.FmtUnsigned)

	//print("dcommontype: %s\n", p);
	ot = dgostringptr(s, ot, p) // string

	// skip pointer to extraType,
	// which follows the rest of this type structure.
	// caller will fill in if needed.
	// otherwise linker will assume 0.
	ot += Widthptr

	ot = dsymptr(s, ot, sptr, 0) // ptrto type
	ot = dsymptr(s, ot, zero, 0) // ptr to zero value
	return ot
}

func typesym(t *Type) *Sym {
	return Pkglookup(Tconv(t, obj.FmtLeft), typepkg)
}

func tracksym(t *Type) *Sym {
	return Pkglookup(Tconv(t.Outer, obj.FmtLeft)+"."+t.Sym.Name, trackpkg)
}

func typelinksym(t *Type) *Sym {
	// %-uT is what the generated Type's string field says.
	// It uses (ambiguous) package names instead of import paths.
	// %-T is the complete, unambiguous type name.
	// We want the types to end up sorted by string field,
	// so use that first in the name, and then add :%-T to
	// disambiguate. We use a tab character as the separator to
	// ensure the types appear sorted by their string field. The
	// names are a little long but they are discarded by the linker
	// and do not end up in the symbol table of the final binary.
	p := Tconv(t, obj.FmtLeft|obj.FmtUnsigned) + "\t" + Tconv(t, obj.FmtLeft)

	s := Pkglookup(p, typelinkpkg)

	//print("typelinksym: %s -> %+S\n", p, s);

	return s
}

func typesymprefix(prefix string, t *Type) *Sym {
	p := prefix + "." + Tconv(t, obj.FmtLeft)
	s := Pkglookup(p, typepkg)

	//print("algsym: %s -> %+S\n", p, s);

	return s
}

func typenamesym(t *Type) *Sym {
	if t == nil || (Isptr[t.Etype] && t.Type == nil) || isideal(t) {
		Fatal("typename %v", Tconv(t, 0))
	}
	s := typesym(t)
	if s.Def == nil {
		n := Nod(ONAME, nil, nil)
		n.Sym = s
		n.Type = Types[TUINT8]
		n.Addable = true
		n.Ullman = 1
		n.Class = PEXTERN
		n.Xoffset = 0
		n.Typecheck = 1
		s.Def = n

		signatlist = list(signatlist, typenod(t))
	}

	return s.Def.Sym
}

func typename(t *Type) *Node {
	s := typenamesym(t)
	n := Nod(OADDR, s.Def, nil)
	n.Type = Ptrto(s.Def.Type)
	n.Addable = true
	n.Ullman = 2
	n.Typecheck = 1
	return n
}

func weaktypesym(t *Type) *Sym {
	p := Tconv(t, obj.FmtLeft)
	s := Pkglookup(p, weaktypepkg)

	//print("weaktypesym: %s -> %+S\n", p, s);

	return s
}

/*
 * Returns 1 if t has a reflexive equality operator.
 * That is, if x==x for all x of type t.
 */
func isreflexive(t *Type) bool {
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
		if Isslice(t) {
			Fatal("slice can't be a map key: %v", Tconv(t, 0))
		}
		return isreflexive(t.Type)

	case TSTRUCT:
		for t1 := t.Type; t1 != nil; t1 = t1.Down {
			if !isreflexive(t1.Type) {
				return false
			}
		}

		return true

	default:
		Fatal("bad type for map key: %v", Tconv(t, 0))
		return false
	}
}

func dtypesym(t *Type) *Sym {
	// Replace byte, rune aliases with real type.
	// They've been separate internally to make error messages
	// better, but we have to merge them in the reflect tables.
	if t == bytetype || t == runetype {
		t = Types[t.Etype]
	}

	if isideal(t) {
		Fatal("dtypesym %v", Tconv(t, 0))
	}

	s := typesym(t)
	if s.Flags&SymSiggen != 0 {
		return s
	}
	s.Flags |= SymSiggen

	// special case (look for runtime below):
	// when compiling package runtime,
	// emit the type structures for int, float, etc.
	tbase := t

	if Isptr[t.Etype] && t.Sym == nil && t.Type.Sym != nil {
		tbase = t.Type
	}
	dupok := 0
	if tbase.Sym == nil {
		dupok = obj.DUPOK
	}

	if compiling_runtime != 0 && (tbase == Types[tbase.Etype] || tbase == bytetype || tbase == runetype || tbase == errortype) { // int, float, etc
		goto ok
	}

	// named types from other files are defined only by those files
	if tbase.Sym != nil && !tbase.Local {
		return s
	}
	if isforw[tbase.Etype] {
		return s
	}

ok:
	ot := 0
	xt := 0
	switch t.Etype {
	default:
		ot = dcommontype(s, ot, t)
		xt = ot - 3*Widthptr

	case TARRAY:
		if t.Bound >= 0 {
			// ../../runtime/type.go:/ArrayType
			s1 := dtypesym(t.Type)

			t2 := typ(TARRAY)
			t2.Type = t.Type
			t2.Bound = -1 // slice
			s2 := dtypesym(t2)
			ot = dcommontype(s, ot, t)
			xt = ot - 3*Widthptr
			ot = dsymptr(s, ot, s1, 0)
			ot = dsymptr(s, ot, s2, 0)
			ot = duintptr(s, ot, uint64(t.Bound))
		} else {
			// ../../runtime/type.go:/SliceType
			s1 := dtypesym(t.Type)

			ot = dcommontype(s, ot, t)
			xt = ot - 3*Widthptr
			ot = dsymptr(s, ot, s1, 0)
		}

		// ../../runtime/type.go:/ChanType
	case TCHAN:
		s1 := dtypesym(t.Type)

		ot = dcommontype(s, ot, t)
		xt = ot - 3*Widthptr
		ot = dsymptr(s, ot, s1, 0)
		ot = duintptr(s, ot, uint64(t.Chan))

	case TFUNC:
		for t1 := getthisx(t).Type; t1 != nil; t1 = t1.Down {
			dtypesym(t1.Type)
		}
		isddd := false
		for t1 := getinargx(t).Type; t1 != nil; t1 = t1.Down {
			isddd = t1.Isddd
			dtypesym(t1.Type)
		}

		for t1 := getoutargx(t).Type; t1 != nil; t1 = t1.Down {
			dtypesym(t1.Type)
		}

		ot = dcommontype(s, ot, t)
		xt = ot - 3*Widthptr
		ot = duint8(s, ot, uint8(bool2int(isddd)))

		// two slice headers: in and out.
		ot = int(Rnd(int64(ot), int64(Widthptr)))

		ot = dsymptr(s, ot, s, ot+2*(Widthptr+2*Widthint))
		n := t.Thistuple + t.Intuple
		ot = duintxx(s, ot, uint64(n), Widthint)
		ot = duintxx(s, ot, uint64(n), Widthint)
		ot = dsymptr(s, ot, s, ot+1*(Widthptr+2*Widthint)+n*Widthptr)
		ot = duintxx(s, ot, uint64(t.Outtuple), Widthint)
		ot = duintxx(s, ot, uint64(t.Outtuple), Widthint)

		// slice data
		for t1 := getthisx(t).Type; t1 != nil; t1 = t1.Down {
			ot = dsymptr(s, ot, dtypesym(t1.Type), 0)
			n++
		}
		for t1 := getinargx(t).Type; t1 != nil; t1 = t1.Down {
			ot = dsymptr(s, ot, dtypesym(t1.Type), 0)
			n++
		}
		for t1 := getoutargx(t).Type; t1 != nil; t1 = t1.Down {
			ot = dsymptr(s, ot, dtypesym(t1.Type), 0)
			n++
		}

	case TINTER:
		m := imethods(t)
		n := 0
		for a := m; a != nil; a = a.link {
			dtypesym(a.type_)
			n++
		}

		// ../../runtime/type.go:/InterfaceType
		ot = dcommontype(s, ot, t)

		xt = ot - 3*Widthptr
		ot = dsymptr(s, ot, s, ot+Widthptr+2*Widthint)
		ot = duintxx(s, ot, uint64(n), Widthint)
		ot = duintxx(s, ot, uint64(n), Widthint)
		for a := m; a != nil; a = a.link {
			// ../../runtime/type.go:/imethod
			ot = dgostringptr(s, ot, a.name)

			ot = dgopkgpath(s, ot, a.pkg)
			ot = dsymptr(s, ot, dtypesym(a.type_), 0)
		}

		// ../../runtime/type.go:/MapType
	case TMAP:
		s1 := dtypesym(t.Down)

		s2 := dtypesym(t.Type)
		s3 := dtypesym(mapbucket(t))
		s4 := dtypesym(hmap(t))
		ot = dcommontype(s, ot, t)
		xt = ot - 3*Widthptr
		ot = dsymptr(s, ot, s1, 0)
		ot = dsymptr(s, ot, s2, 0)
		ot = dsymptr(s, ot, s3, 0)
		ot = dsymptr(s, ot, s4, 0)
		if t.Down.Width > MAXKEYSIZE {
			ot = duint8(s, ot, uint8(Widthptr))
			ot = duint8(s, ot, 1) // indirect
		} else {
			ot = duint8(s, ot, uint8(t.Down.Width))
			ot = duint8(s, ot, 0) // not indirect
		}

		if t.Type.Width > MAXVALSIZE {
			ot = duint8(s, ot, uint8(Widthptr))
			ot = duint8(s, ot, 1) // indirect
		} else {
			ot = duint8(s, ot, uint8(t.Type.Width))
			ot = duint8(s, ot, 0) // not indirect
		}

		ot = duint16(s, ot, uint16(mapbucket(t).Width))
		ot = duint8(s, ot, uint8(bool2int(isreflexive(t.Down))))

	case TPTR32, TPTR64:
		if t.Type.Etype == TANY {
			// ../../runtime/type.go:/UnsafePointerType
			ot = dcommontype(s, ot, t)

			break
		}

		// ../../runtime/type.go:/PtrType
		s1 := dtypesym(t.Type)

		ot = dcommontype(s, ot, t)
		xt = ot - 3*Widthptr
		ot = dsymptr(s, ot, s1, 0)

		// ../../runtime/type.go:/StructType
	// for security, only the exported fields.
	case TSTRUCT:
		n := 0

		for t1 := t.Type; t1 != nil; t1 = t1.Down {
			dtypesym(t1.Type)
			n++
		}

		ot = dcommontype(s, ot, t)
		xt = ot - 3*Widthptr
		ot = dsymptr(s, ot, s, ot+Widthptr+2*Widthint)
		ot = duintxx(s, ot, uint64(n), Widthint)
		ot = duintxx(s, ot, uint64(n), Widthint)
		for t1 := t.Type; t1 != nil; t1 = t1.Down {
			// ../../runtime/type.go:/structField
			if t1.Sym != nil && t1.Embedded == 0 {
				ot = dgostringptr(s, ot, t1.Sym.Name)
				if exportname(t1.Sym.Name) {
					ot = dgostringptr(s, ot, "")
				} else {
					ot = dgopkgpath(s, ot, t1.Sym.Pkg)
				}
			} else {
				ot = dgostringptr(s, ot, "")
				if t1.Type.Sym != nil && t1.Type.Sym.Pkg == builtinpkg {
					ot = dgopkgpath(s, ot, localpkg)
				} else {
					ot = dgostringptr(s, ot, "")
				}
			}

			ot = dsymptr(s, ot, dtypesym(t1.Type), 0)
			ot = dgostrlitptr(s, ot, t1.Note)
			ot = duintptr(s, ot, uint64(t1.Width)) // field offset
		}
	}

	ot = dextratype(s, ot, t, xt)
	ggloblsym(s, int32(ot), int8(dupok|obj.RODATA))

	// generate typelink.foo pointing at s = type.foo.
	// The linker will leave a table of all the typelinks for
	// types in the binary, so reflect can find them.
	// We only need the link for unnamed composites that
	// we want be able to find.
	if t.Sym == nil {
		switch t.Etype {
		case TARRAY, TCHAN, TFUNC, TMAP:
			slink := typelinksym(t)
			dsymptr(slink, 0, s, 0)
			ggloblsym(slink, int32(Widthptr), int8(dupok|obj.RODATA))
		}
	}

	return s
}

func dumptypestructs() {
	var n *Node

	// copy types from externdcl list to signatlist
	for l := externdcl; l != nil; l = l.Next {
		n = l.N
		if n.Op != OTYPE {
			continue
		}
		signatlist = list(signatlist, n)
	}

	// process signatlist
	var t *Type
	for l := signatlist; l != nil; l = l.Next {
		n = l.N
		if n.Op != OTYPE {
			continue
		}
		t = n.Type
		dtypesym(t)
		if t.Sym != nil {
			dtypesym(Ptrto(t))
		}
	}

	// generate import strings for imported packages
	for _, p := range pkgs {
		if p.Direct != 0 {
			dimportpath(p)
		}
	}

	// do basic types if compiling package runtime.
	// they have to be in at least one package,
	// and runtime is always loaded implicitly,
	// so this is as good as any.
	// another possible choice would be package main,
	// but using runtime means fewer copies in .6 files.
	if compiling_runtime != 0 {
		for i := 1; i <= TBOOL; i++ {
			dtypesym(Ptrto(Types[i]))
		}
		dtypesym(Ptrto(Types[TSTRING]))
		dtypesym(Ptrto(Types[TUNSAFEPTR]))

		// emit type structs for error and func(error) string.
		// The latter is the type of an auto-generated wrapper.
		dtypesym(Ptrto(errortype))

		dtypesym(functype(nil, list1(Nod(ODCLFIELD, nil, typenod(errortype))), list1(Nod(ODCLFIELD, nil, typenod(Types[TSTRING])))))

		// add paths for runtime and main, which 6l imports implicitly.
		dimportpath(Runtimepkg)

		if flag_race != 0 {
			dimportpath(racepkg)
		}
		dimportpath(mkpkg("main"))
	}
}

func dalgsym(t *Type) *Sym {
	var s *Sym
	var hashfunc *Sym
	var eqfunc *Sym

	// dalgsym is only called for a type that needs an algorithm table,
	// which implies that the type is comparable (or else it would use ANOEQ).

	if algtype(t) == AMEM {
		// we use one algorithm table for all AMEM types of a given size
		p := fmt.Sprintf(".alg%d", t.Width)

		s = Pkglookup(p, typepkg)

		if s.Flags&SymAlgGen != 0 {
			return s
		}
		s.Flags |= SymAlgGen

		// make hash closure
		p = fmt.Sprintf(".hashfunc%d", t.Width)

		hashfunc = Pkglookup(p, typepkg)

		ot := 0
		ot = dsymptr(hashfunc, ot, Pkglookup("memhash_varlen", Runtimepkg), 0)
		ot = duintxx(hashfunc, ot, uint64(t.Width), Widthptr) // size encoded in closure
		ggloblsym(hashfunc, int32(ot), obj.DUPOK|obj.RODATA)

		// make equality closure
		p = fmt.Sprintf(".eqfunc%d", t.Width)

		eqfunc = Pkglookup(p, typepkg)

		ot = 0
		ot = dsymptr(eqfunc, ot, Pkglookup("memequal_varlen", Runtimepkg), 0)
		ot = duintxx(eqfunc, ot, uint64(t.Width), Widthptr)
		ggloblsym(eqfunc, int32(ot), obj.DUPOK|obj.RODATA)
	} else {
		// generate an alg table specific to this type
		s = typesymprefix(".alg", t)

		hash := typesymprefix(".hash", t)
		eq := typesymprefix(".eq", t)
		hashfunc = typesymprefix(".hashfunc", t)
		eqfunc = typesymprefix(".eqfunc", t)

		genhash(hash, t)
		geneq(eq, t)

		// make Go funcs (closures) for calling hash and equal from Go
		dsymptr(hashfunc, 0, hash, 0)

		ggloblsym(hashfunc, int32(Widthptr), obj.DUPOK|obj.RODATA)
		dsymptr(eqfunc, 0, eq, 0)
		ggloblsym(eqfunc, int32(Widthptr), obj.DUPOK|obj.RODATA)
	}

	// ../../runtime/alg.go:/typeAlg
	ot := 0

	ot = dsymptr(s, ot, hashfunc, 0)
	ot = dsymptr(s, ot, eqfunc, 0)
	ggloblsym(s, int32(ot), obj.DUPOK|obj.RODATA)
	return s
}

func usegcprog(t *Type) bool {
	if !haspointers(t) {
		return false
	}
	if t.Width == BADWIDTH {
		dowidth(t)
	}

	// Calculate size of the unrolled GC mask.
	nptr := (t.Width + int64(Widthptr) - 1) / int64(Widthptr)

	size := nptr
	if size%2 != 0 {
		size *= 2 // repeated
	}
	size = size * obj.GcBits / 8 // 4 bits per word

	// Decide whether to use unrolled GC mask or GC program.
	// We could use a more elaborate condition, but this seems to work well in practice.
	// For small objects GC program can't give significant reduction.
	// While large objects usually contain arrays; and even if it don't
	// the program uses 2-bits per word while mask uses 4-bits per word,
	// so the program is still smaller.
	return size > int64(2*Widthptr)
}

// Generates sparse GC bitmask (4 bits per word).
func gengcmask(t *Type, gcmask []byte) {
	for i := int64(0); i < 16; i++ {
		gcmask[i] = 0
	}
	if !haspointers(t) {
		return
	}

	// Generate compact mask as stacks use.
	xoffset := int64(0)

	vec := bvalloc(2 * int32(Widthptr) * 8)
	twobitwalktype1(t, &xoffset, vec)

	// Unfold the mask for the GC bitmap format:
	// 4 bits per word, 2 high bits encode pointer info.
	pos := gcmask

	nptr := (t.Width + int64(Widthptr) - 1) / int64(Widthptr)
	half := false

	// If number of words is odd, repeat the mask.
	// This makes simpler handling of arrays in runtime.
	var i int64
	var bits uint8
	for j := int64(0); j <= (nptr % 2); j++ {
		for i = 0; i < nptr; i++ {
			bits = uint8(bvget(vec, int32(i*obj.BitsPerPointer)) | bvget(vec, int32(i*obj.BitsPerPointer+1))<<1)

			// Some fake types (e.g. Hmap) has missing fileds.
			// twobitwalktype1 generates BitsDead for that holes,
			// replace BitsDead with BitsScalar.
			if bits == obj.BitsDead {
				bits = obj.BitsScalar
			}
			bits <<= 2
			if half {
				bits <<= 4
			}
			pos[0] |= byte(bits)
			half = !half
			if !half {
				pos = pos[1:]
			}
		}
	}
}

// Helper object for generation of GC programs.
type ProgGen struct {
	s        *Sym
	datasize int32
	data     [256 / obj.PointersPerByte]uint8
	ot       int64
}

func proggeninit(g *ProgGen, s *Sym) {
	g.s = s
	g.datasize = 0
	g.ot = 0
	g.data = [256 / obj.PointersPerByte]uint8{}
}

func proggenemit(g *ProgGen, v uint8) {
	g.ot = int64(duint8(g.s, int(g.ot), v))
}

// Emits insData block from g->data.
func proggendataflush(g *ProgGen) {
	if g.datasize == 0 {
		return
	}
	proggenemit(g, obj.InsData)
	proggenemit(g, uint8(g.datasize))
	s := (g.datasize + obj.PointersPerByte - 1) / obj.PointersPerByte
	for i := int32(0); i < s; i++ {
		proggenemit(g, g.data[i])
	}
	g.datasize = 0
	g.data = [256 / obj.PointersPerByte]uint8{}
}

func proggendata(g *ProgGen, d uint8) {
	g.data[g.datasize/obj.PointersPerByte] |= d << uint((g.datasize%obj.PointersPerByte)*obj.BitsPerPointer)
	g.datasize++
	if g.datasize == 255 {
		proggendataflush(g)
	}
}

// Skip v bytes due to alignment, etc.
func proggenskip(g *ProgGen, off int64, v int64) {
	for i := off; i < off+v; i++ {
		if (i % int64(Widthptr)) == 0 {
			proggendata(g, obj.BitsScalar)
		}
	}
}

// Emit insArray instruction.
func proggenarray(g *ProgGen, len int64) {
	proggendataflush(g)
	proggenemit(g, obj.InsArray)
	for i := int32(0); i < int32(Widthptr); i, len = i+1, len>>8 {
		proggenemit(g, uint8(len))
	}
}

func proggenarrayend(g *ProgGen) {
	proggendataflush(g)
	proggenemit(g, obj.InsArrayEnd)
}

func proggenfini(g *ProgGen) int64 {
	proggendataflush(g)
	proggenemit(g, obj.InsEnd)
	return g.ot
}

// Generates GC program for large types.
func gengcprog(t *Type, pgc0 **Sym, pgc1 **Sym) {
	nptr := (t.Width + int64(Widthptr) - 1) / int64(Widthptr)
	size := nptr
	if size%2 != 0 {
		size *= 2 // repeated twice
	}
	size = size * obj.PointersPerByte / 8 // 4 bits per word
	size++                                // unroll flag in the beginning, used by runtime (see runtime.markallocated)

	// emity space in BSS for unrolled program
	*pgc0 = nil

	// Don't generate it if it's too large, runtime will unroll directly into GC bitmap.
	if size <= obj.MaxGCMask {
		gc0 := typesymprefix(".gc", t)
		ggloblsym(gc0, int32(size), obj.DUPOK|obj.NOPTR)
		*pgc0 = gc0
	}

	// program in RODATA
	gc1 := typesymprefix(".gcprog", t)

	var g ProgGen
	proggeninit(&g, gc1)
	xoffset := int64(0)
	gengcprog1(&g, t, &xoffset)
	ot := proggenfini(&g)
	ggloblsym(gc1, int32(ot), obj.DUPOK|obj.RODATA)
	*pgc1 = gc1
}

// Recursively walks type t and writes GC program into g.
func gengcprog1(g *ProgGen, t *Type, xoffset *int64) {
	switch t.Etype {
	case TINT8,
		TUINT8,
		TINT16,
		TUINT16,
		TINT32,
		TUINT32,
		TINT64,
		TUINT64,
		TINT,
		TUINT,
		TUINTPTR,
		TBOOL,
		TFLOAT32,
		TFLOAT64,
		TCOMPLEX64,
		TCOMPLEX128:
		proggenskip(g, *xoffset, t.Width)
		*xoffset += t.Width

	case TPTR32,
		TPTR64,
		TUNSAFEPTR,
		TFUNC,
		TCHAN,
		TMAP:
		proggendata(g, obj.BitsPointer)
		*xoffset += t.Width

	case TSTRING:
		proggendata(g, obj.BitsPointer)
		proggendata(g, obj.BitsScalar)
		*xoffset += t.Width

		// Assuming IfacePointerOnly=1.
	case TINTER:
		proggendata(g, obj.BitsPointer)

		proggendata(g, obj.BitsPointer)
		*xoffset += t.Width

	case TARRAY:
		if Isslice(t) {
			proggendata(g, obj.BitsPointer)
			proggendata(g, obj.BitsScalar)
			proggendata(g, obj.BitsScalar)
		} else {
			t1 := t.Type
			if t1.Width == 0 {
			}
			// ignore
			if t.Bound <= 1 || t.Bound*t1.Width < int64(32*Widthptr) {
				for i := int64(0); i < t.Bound; i++ {
					gengcprog1(g, t1, xoffset)
				}
			} else if !haspointers(t1) {
				n := t.Width
				n -= -*xoffset & (int64(Widthptr) - 1) // skip to next ptr boundary
				proggenarray(g, (n+int64(Widthptr)-1)/int64(Widthptr))
				proggendata(g, obj.BitsScalar)
				proggenarrayend(g)
				*xoffset -= (n+int64(Widthptr)-1)/int64(Widthptr)*int64(Widthptr) - t.Width
			} else {
				proggenarray(g, t.Bound)
				gengcprog1(g, t1, xoffset)
				*xoffset += (t.Bound - 1) * t1.Width
				proggenarrayend(g)
			}
		}

	case TSTRUCT:
		o := int64(0)
		var fieldoffset int64
		for t1 := t.Type; t1 != nil; t1 = t1.Down {
			fieldoffset = t1.Width
			proggenskip(g, *xoffset, fieldoffset-o)
			*xoffset += fieldoffset - o
			gengcprog1(g, t1.Type, xoffset)
			o = fieldoffset + t1.Type.Width
		}

		proggenskip(g, *xoffset, t.Width-o)
		*xoffset += t.Width - o

	default:
		Fatal("gengcprog1: unexpected type, %v", Tconv(t, 0))
	}
}

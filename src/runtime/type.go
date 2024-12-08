// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Runtime type representation.

package runtime

import (
	"internal/abi"
	"internal/goarch"
	"internal/goexperiment"
	"internal/runtime/atomic"
	"unsafe"
)

type nameOff = abi.NameOff
type typeOff = abi.TypeOff
type textOff = abi.TextOff

type _type = abi.Type

// rtype is a wrapper that allows us to define additional methods.
type rtype struct {
	*abi.Type // embedding is okay here (unlike reflect) because none of this is public
}

func (t rtype) string() string {
	s := t.nameOff(t.Str).Name()
	if t.TFlag&abi.TFlagExtraStar != 0 {
		return s[1:]
	}
	return s
}

func (t rtype) uncommon() *uncommontype {
	return t.Uncommon()
}

func (t rtype) name() string {
	if t.TFlag&abi.TFlagNamed == 0 {
		return ""
	}
	s := t.string()
	i := len(s) - 1
	sqBrackets := 0
	for i >= 0 && (s[i] != '.' || sqBrackets != 0) {
		switch s[i] {
		case ']':
			sqBrackets++
		case '[':
			sqBrackets--
		}
		i--
	}
	return s[i+1:]
}

// pkgpath returns the path of the package where t was defined, if
// available. This is not the same as the reflect package's PkgPath
// method, in that it returns the package path for struct and interface
// types, not just named types.
func (t rtype) pkgpath() string {
	if u := t.uncommon(); u != nil {
		return t.nameOff(u.PkgPath).Name()
	}
	switch t.Kind_ & abi.KindMask {
	case abi.Struct:
		st := (*structtype)(unsafe.Pointer(t.Type))
		return st.PkgPath.Name()
	case abi.Interface:
		it := (*interfacetype)(unsafe.Pointer(t.Type))
		return it.PkgPath.Name()
	}
	return ""
}

// getGCMask returns the pointer/nonpointer bitmask for type t.
//
// nosplit because it is used during write barriers and must not be preempted.
//
//go:nosplit
func getGCMask(t *_type) *byte {
	if t.TFlag&abi.TFlagGCMaskOnDemand != 0 {
		// Split the rest into getGCMaskOnDemand so getGCMask itself is inlineable.
		return getGCMaskOnDemand(t)
	}
	return t.GCData
}

// inProgress is a byte whose address is a sentinel indicating that
// some thread is currently building the GC bitmask for a type.
var inProgress byte

// nosplit because it is used during write barriers and must not be preempted.
//
//go:nosplit
func getGCMaskOnDemand(t *_type) *byte {
	// For large types, GCData doesn't point directly to a bitmask.
	// Instead it points to a pointer to a bitmask, and the runtime
	// is responsible for (on first use) creating the bitmask and
	// storing a pointer to it in that slot.
	// TODO: we could use &t.GCData as the slot, but types are
	// in read-only memory currently.
	addr := unsafe.Pointer(t.GCData)

	for {
		p := (*byte)(atomic.Loadp(addr))
		switch p {
		default: // Already built.
			return p
		case &inProgress: // Someone else is currently building it.
			// Just wait until the builder is done.
			// We can't block here, so spinning while having
			// the OS thread yield is about the best we can do.
			osyield()
			continue
		case nil: // Not built yet.
			// Attempt to get exclusive access to build it.
			if !atomic.Casp1((*unsafe.Pointer)(addr), nil, unsafe.Pointer(&inProgress)) {
				continue
			}

			// Build gcmask for this type.
			bytes := goarch.PtrSize * divRoundUp(t.PtrBytes/goarch.PtrSize, 8*goarch.PtrSize)
			p = (*byte)(persistentalloc(bytes, goarch.PtrSize, &memstats.other_sys))
			systemstack(func() {
				buildGCMask(t, bitCursor{ptr: p, n: 0})
			})

			// Store the newly-built gcmask for future callers.
			atomic.StorepNoWB(addr, unsafe.Pointer(p))
			return p
		}
	}
}

// A bitCursor is a simple cursor to memory to which we
// can write a set of bits.
type bitCursor struct {
	ptr *byte   // base of region
	n   uintptr // cursor points to bit n of region
}

// Write to b cnt bits starting at bit 0 of data.
// Requires cnt>0.
func (b bitCursor) write(data *byte, cnt uintptr) {
	// Starting byte for writing.
	p := addb(b.ptr, b.n/8)

	// Note: if we're starting halfway through a byte, we load the
	// existing lower bits so we don't clobber them.
	n := b.n % 8                    // # of valid bits in buf
	buf := uintptr(*p) & (1<<n - 1) // buffered bits to start

	// Work 8 bits at a time.
	for cnt > 8 {
		// Read 8 more bits, now buf has 8-15 valid bits in it.
		buf |= uintptr(*data) << n
		n += 8
		data = addb(data, 1)
		cnt -= 8
		// Write 8 of the buffered bits out.
		*p = byte(buf)
		buf >>= 8
		n -= 8
		p = addb(p, 1)
	}
	// Read remaining bits.
	buf |= (uintptr(*data) & (1<<cnt - 1)) << n
	n += cnt

	// Flush remaining bits.
	if n > 8 {
		*p = byte(buf)
		buf >>= 8
		n -= 8
		p = addb(p, 1)
	}
	*p &^= 1<<n - 1
	*p |= byte(buf)
}

func (b bitCursor) offset(cnt uintptr) bitCursor {
	return bitCursor{ptr: b.ptr, n: b.n + cnt}
}

// buildGCMask writes the ptr/nonptr bitmap for t to dst.
// t must have a pointer.
func buildGCMask(t *_type, dst bitCursor) {
	// Note: we want to avoid a situation where buildGCMask gets into a
	// very deep recursion, because M stacks are fixed size and pretty small
	// (16KB). We do that by ensuring that any recursive
	// call operates on a type at most half the size of its parent.
	// Thus, the recursive chain can be at most 64 calls deep (on a
	// 64-bit machine).
	// Recursion is avoided by using a "tail call" (jumping to the
	// "top" label) for any recursive call with a large subtype.
top:
	if t.PtrBytes == 0 {
		throw("pointerless type")
	}
	if t.TFlag&abi.TFlagGCMaskOnDemand == 0 {
		// copy t.GCData to dst
		dst.write(t.GCData, t.PtrBytes/goarch.PtrSize)
		return
	}
	// The above case should handle all kinds except
	// possibly arrays and structs.
	switch t.Kind() {
	case abi.Array:
		a := t.ArrayType()
		if a.Len == 1 {
			// Avoid recursive call for element type that
			// isn't smaller than the parent type.
			t = a.Elem
			goto top
		}
		e := a.Elem
		for i := uintptr(0); i < a.Len; i++ {
			buildGCMask(e, dst)
			dst = dst.offset(e.Size_ / goarch.PtrSize)
		}
	case abi.Struct:
		s := t.StructType()
		var bigField abi.StructField
		for _, f := range s.Fields {
			ft := f.Typ
			if !ft.Pointers() {
				continue
			}
			if ft.Size_ > t.Size_/2 {
				// Avoid recursive call for field type that
				// is larger than half of the parent type.
				// There can be only one.
				bigField = f
				continue
			}
			buildGCMask(ft, dst.offset(f.Offset/goarch.PtrSize))
		}
		if bigField.Typ != nil {
			// Note: this case causes bits to be written out of order.
			t = bigField.Typ
			dst = dst.offset(bigField.Offset / goarch.PtrSize)
			goto top
		}
	default:
		throw("unexpected kind")
	}
}

// reflectOffs holds type offsets defined at run time by the reflect package.
//
// When a type is defined at run time, its *rtype data lives on the heap.
// There are a wide range of possible addresses the heap may use, that
// may not be representable as a 32-bit offset. Moreover the GC may
// one day start moving heap memory, in which case there is no stable
// offset that can be defined.
//
// To provide stable offsets, we add pin *rtype objects in a global map
// and treat the offset as an identifier. We use negative offsets that
// do not overlap with any compile-time module offsets.
//
// Entries are created by reflect.addReflectOff.
var reflectOffs struct {
	lock mutex
	next int32
	m    map[int32]unsafe.Pointer
	minv map[unsafe.Pointer]int32
}

func reflectOffsLock() {
	lock(&reflectOffs.lock)
	if raceenabled {
		raceacquire(unsafe.Pointer(&reflectOffs.lock))
	}
}

func reflectOffsUnlock() {
	if raceenabled {
		racerelease(unsafe.Pointer(&reflectOffs.lock))
	}
	unlock(&reflectOffs.lock)
}

func resolveNameOff(ptrInModule unsafe.Pointer, off nameOff) name {
	if off == 0 {
		return name{}
	}
	base := uintptr(ptrInModule)
	for md := &firstmoduledata; md != nil; md = md.next {
		if base >= md.types && base < md.etypes {
			res := md.types + uintptr(off)
			if res > md.etypes {
				println("runtime: nameOff", hex(off), "out of range", hex(md.types), "-", hex(md.etypes))
				throw("runtime: name offset out of range")
			}
			return name{Bytes: (*byte)(unsafe.Pointer(res))}
		}
	}

	// No module found. see if it is a run time name.
	reflectOffsLock()
	res, found := reflectOffs.m[int32(off)]
	reflectOffsUnlock()
	if !found {
		println("runtime: nameOff", hex(off), "base", hex(base), "not in ranges:")
		for next := &firstmoduledata; next != nil; next = next.next {
			println("\ttypes", hex(next.types), "etypes", hex(next.etypes))
		}
		throw("runtime: name offset base pointer out of range")
	}
	return name{Bytes: (*byte)(res)}
}

func (t rtype) nameOff(off nameOff) name {
	return resolveNameOff(unsafe.Pointer(t.Type), off)
}

func resolveTypeOff(ptrInModule unsafe.Pointer, off typeOff) *_type {
	if off == 0 || off == -1 {
		// -1 is the sentinel value for unreachable code.
		// See cmd/link/internal/ld/data.go:relocsym.
		return nil
	}
	base := uintptr(ptrInModule)
	var md *moduledata
	for next := &firstmoduledata; next != nil; next = next.next {
		if base >= next.types && base < next.etypes {
			md = next
			break
		}
	}
	if md == nil {
		reflectOffsLock()
		res := reflectOffs.m[int32(off)]
		reflectOffsUnlock()
		if res == nil {
			println("runtime: typeOff", hex(off), "base", hex(base), "not in ranges:")
			for next := &firstmoduledata; next != nil; next = next.next {
				println("\ttypes", hex(next.types), "etypes", hex(next.etypes))
			}
			throw("runtime: type offset base pointer out of range")
		}
		return (*_type)(res)
	}
	if t := md.typemap[off]; t != nil {
		return t
	}
	res := md.types + uintptr(off)
	if res > md.etypes {
		println("runtime: typeOff", hex(off), "out of range", hex(md.types), "-", hex(md.etypes))
		throw("runtime: type offset out of range")
	}
	return (*_type)(unsafe.Pointer(res))
}

func (t rtype) typeOff(off typeOff) *_type {
	return resolveTypeOff(unsafe.Pointer(t.Type), off)
}

func (t rtype) textOff(off textOff) unsafe.Pointer {
	if off == -1 {
		// -1 is the sentinel value for unreachable code.
		// See cmd/link/internal/ld/data.go:relocsym.
		return unsafe.Pointer(abi.FuncPCABIInternal(unreachableMethod))
	}
	base := uintptr(unsafe.Pointer(t.Type))
	var md *moduledata
	for next := &firstmoduledata; next != nil; next = next.next {
		if base >= next.types && base < next.etypes {
			md = next
			break
		}
	}
	if md == nil {
		reflectOffsLock()
		res := reflectOffs.m[int32(off)]
		reflectOffsUnlock()
		if res == nil {
			println("runtime: textOff", hex(off), "base", hex(base), "not in ranges:")
			for next := &firstmoduledata; next != nil; next = next.next {
				println("\ttypes", hex(next.types), "etypes", hex(next.etypes))
			}
			throw("runtime: text offset base pointer out of range")
		}
		return res
	}
	res := md.textAddr(uint32(off))
	return unsafe.Pointer(res)
}

type uncommontype = abi.UncommonType

type interfacetype = abi.InterfaceType

type arraytype = abi.ArrayType

type chantype = abi.ChanType

type slicetype = abi.SliceType

type functype = abi.FuncType

type ptrtype = abi.PtrType

type name = abi.Name

type structtype = abi.StructType

func pkgPath(n name) string {
	if n.Bytes == nil || *n.Data(0)&(1<<2) == 0 {
		return ""
	}
	i, l := n.ReadVarint(1)
	off := 1 + i + l
	if *n.Data(0)&(1<<1) != 0 {
		i2, l2 := n.ReadVarint(off)
		off += i2 + l2
	}
	var nameOff nameOff
	copy((*[4]byte)(unsafe.Pointer(&nameOff))[:], (*[4]byte)(unsafe.Pointer(n.Data(off)))[:])
	pkgPathName := resolveNameOff(unsafe.Pointer(n.Bytes), nameOff)
	return pkgPathName.Name()
}

// typelinksinit scans the types from extra modules and builds the
// moduledata typemap used to de-duplicate type pointers.
func typelinksinit() {
	if firstmoduledata.next == nil {
		return
	}
	typehash := make(map[uint32][]*_type, len(firstmoduledata.typelinks))

	modules := activeModules()
	prev := modules[0]
	for _, md := range modules[1:] {
		// Collect types from the previous module into typehash.
	collect:
		for _, tl := range prev.typelinks {
			var t *_type
			if prev.typemap == nil {
				t = (*_type)(unsafe.Pointer(prev.types + uintptr(tl)))
			} else {
				t = prev.typemap[typeOff(tl)]
			}
			// Add to typehash if not seen before.
			tlist := typehash[t.Hash]
			for _, tcur := range tlist {
				if tcur == t {
					continue collect
				}
			}
			typehash[t.Hash] = append(tlist, t)
		}

		if md.typemap == nil {
			// If any of this module's typelinks match a type from a
			// prior module, prefer that prior type by adding the offset
			// to this module's typemap.
			tm := make(map[typeOff]*_type, len(md.typelinks))
			pinnedTypemaps = append(pinnedTypemaps, tm)
			md.typemap = tm
			for _, tl := range md.typelinks {
				t := (*_type)(unsafe.Pointer(md.types + uintptr(tl)))
				for _, candidate := range typehash[t.Hash] {
					seen := map[_typePair]struct{}{}
					if typesEqual(t, candidate, seen) {
						t = candidate
						break
					}
				}
				md.typemap[typeOff(tl)] = t
			}
		}

		prev = md
	}
}

type _typePair struct {
	t1 *_type
	t2 *_type
}

func toRType(t *abi.Type) rtype {
	return rtype{t}
}

// typesEqual reports whether two types are equal.
//
// Everywhere in the runtime and reflect packages, it is assumed that
// there is exactly one *_type per Go type, so that pointer equality
// can be used to test if types are equal. There is one place that
// breaks this assumption: buildmode=shared. In this case a type can
// appear as two different pieces of memory. This is hidden from the
// runtime and reflect package by the per-module typemap built in
// typelinksinit. It uses typesEqual to map types from later modules
// back into earlier ones.
//
// Only typelinksinit needs this function.
func typesEqual(t, v *_type, seen map[_typePair]struct{}) bool {
	tp := _typePair{t, v}
	if _, ok := seen[tp]; ok {
		return true
	}

	// mark these types as seen, and thus equivalent which prevents an infinite loop if
	// the two types are identical, but recursively defined and loaded from
	// different modules
	seen[tp] = struct{}{}

	if t == v {
		return true
	}
	kind := t.Kind_ & abi.KindMask
	if kind != v.Kind_&abi.KindMask {
		return false
	}
	rt, rv := toRType(t), toRType(v)
	if rt.string() != rv.string() {
		return false
	}
	ut := t.Uncommon()
	uv := v.Uncommon()
	if ut != nil || uv != nil {
		if ut == nil || uv == nil {
			return false
		}
		pkgpatht := rt.nameOff(ut.PkgPath).Name()
		pkgpathv := rv.nameOff(uv.PkgPath).Name()
		if pkgpatht != pkgpathv {
			return false
		}
	}
	if abi.Bool <= kind && kind <= abi.Complex128 {
		return true
	}
	switch kind {
	case abi.String, abi.UnsafePointer:
		return true
	case abi.Array:
		at := (*arraytype)(unsafe.Pointer(t))
		av := (*arraytype)(unsafe.Pointer(v))
		return typesEqual(at.Elem, av.Elem, seen) && at.Len == av.Len
	case abi.Chan:
		ct := (*chantype)(unsafe.Pointer(t))
		cv := (*chantype)(unsafe.Pointer(v))
		return ct.Dir == cv.Dir && typesEqual(ct.Elem, cv.Elem, seen)
	case abi.Func:
		ft := (*functype)(unsafe.Pointer(t))
		fv := (*functype)(unsafe.Pointer(v))
		if ft.OutCount != fv.OutCount || ft.InCount != fv.InCount {
			return false
		}
		tin, vin := ft.InSlice(), fv.InSlice()
		for i := 0; i < len(tin); i++ {
			if !typesEqual(tin[i], vin[i], seen) {
				return false
			}
		}
		tout, vout := ft.OutSlice(), fv.OutSlice()
		for i := 0; i < len(tout); i++ {
			if !typesEqual(tout[i], vout[i], seen) {
				return false
			}
		}
		return true
	case abi.Interface:
		it := (*interfacetype)(unsafe.Pointer(t))
		iv := (*interfacetype)(unsafe.Pointer(v))
		if it.PkgPath.Name() != iv.PkgPath.Name() {
			return false
		}
		if len(it.Methods) != len(iv.Methods) {
			return false
		}
		for i := range it.Methods {
			tm := &it.Methods[i]
			vm := &iv.Methods[i]
			// Note the mhdr array can be relocated from
			// another module. See #17724.
			tname := resolveNameOff(unsafe.Pointer(tm), tm.Name)
			vname := resolveNameOff(unsafe.Pointer(vm), vm.Name)
			if tname.Name() != vname.Name() {
				return false
			}
			if pkgPath(tname) != pkgPath(vname) {
				return false
			}
			tityp := resolveTypeOff(unsafe.Pointer(tm), tm.Typ)
			vityp := resolveTypeOff(unsafe.Pointer(vm), vm.Typ)
			if !typesEqual(tityp, vityp, seen) {
				return false
			}
		}
		return true
	case abi.Map:
		if goexperiment.SwissMap {
			mt := (*abi.SwissMapType)(unsafe.Pointer(t))
			mv := (*abi.SwissMapType)(unsafe.Pointer(v))
			return typesEqual(mt.Key, mv.Key, seen) && typesEqual(mt.Elem, mv.Elem, seen)
		}
		mt := (*abi.OldMapType)(unsafe.Pointer(t))
		mv := (*abi.OldMapType)(unsafe.Pointer(v))
		return typesEqual(mt.Key, mv.Key, seen) && typesEqual(mt.Elem, mv.Elem, seen)
	case abi.Pointer:
		pt := (*ptrtype)(unsafe.Pointer(t))
		pv := (*ptrtype)(unsafe.Pointer(v))
		return typesEqual(pt.Elem, pv.Elem, seen)
	case abi.Slice:
		st := (*slicetype)(unsafe.Pointer(t))
		sv := (*slicetype)(unsafe.Pointer(v))
		return typesEqual(st.Elem, sv.Elem, seen)
	case abi.Struct:
		st := (*structtype)(unsafe.Pointer(t))
		sv := (*structtype)(unsafe.Pointer(v))
		if len(st.Fields) != len(sv.Fields) {
			return false
		}
		if st.PkgPath.Name() != sv.PkgPath.Name() {
			return false
		}
		for i := range st.Fields {
			tf := &st.Fields[i]
			vf := &sv.Fields[i]
			if tf.Name.Name() != vf.Name.Name() {
				return false
			}
			if !typesEqual(tf.Typ, vf.Typ, seen) {
				return false
			}
			if tf.Name.Tag() != vf.Name.Tag() {
				return false
			}
			if tf.Offset != vf.Offset {
				return false
			}
			if tf.Name.IsEmbedded() != vf.Name.IsEmbedded() {
				return false
			}
		}
		return true
	default:
		println("runtime: impossible type kind", kind)
		throw("runtime: impossible type kind")
		return false
	}
}

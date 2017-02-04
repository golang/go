// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"runtime/internal/atomic"
	"runtime/internal/sys"
	"unsafe"
)

const (
	hashSize = 1009
)

var (
	ifaceLock mutex // lock for accessing hash
	hash      [hashSize]*itab
)

func itabhash(inter *interfacetype, typ *_type) uint32 {
	// compiler has provided some good hash codes for us.
	h := inter.typ.hash
	h += 17 * typ.hash
	// TODO(rsc): h += 23 * x.mhash ?
	return h % hashSize
}

func getitab(inter *interfacetype, typ *_type, canfail bool) *itab {
	if len(inter.mhdr) == 0 {
		throw("internal error - misuse of itab")
	}

	// easy case
	if typ.tflag&tflagUncommon == 0 {
		if canfail {
			return nil
		}
		name := inter.typ.nameOff(inter.mhdr[0].name)
		panic(&TypeAssertionError{"", typ.string(), inter.typ.string(), name.name()})
	}

	h := itabhash(inter, typ)

	// look twice - once without lock, once with.
	// common case will be no lock contention.
	var m *itab
	var locked int
	for locked = 0; locked < 2; locked++ {
		if locked != 0 {
			lock(&ifaceLock)
		}
		for m = (*itab)(atomic.Loadp(unsafe.Pointer(&hash[h]))); m != nil; m = m.link {
			if m.inter == inter && m._type == typ {
				if m.bad {
					if !canfail {
						// this can only happen if the conversion
						// was already done once using the , ok form
						// and we have a cached negative result.
						// the cached result doesn't record which
						// interface function was missing, so try
						// adding the itab again, which will throw an error.
						additab(m, locked != 0, false)
					}
					m = nil
				}
				if locked != 0 {
					unlock(&ifaceLock)
				}
				return m
			}
		}
	}

	m = (*itab)(persistentalloc(unsafe.Sizeof(itab{})+uintptr(len(inter.mhdr)-1)*sys.PtrSize, 0, &memstats.other_sys))
	m.inter = inter
	m._type = typ
	additab(m, true, canfail)
	unlock(&ifaceLock)
	if m.bad {
		return nil
	}
	return m
}

func additab(m *itab, locked, canfail bool) {
	inter := m.inter
	typ := m._type
	x := typ.uncommon()

	// both inter and typ have method sorted by name,
	// and interface names are unique,
	// so can iterate over both in lock step;
	// the loop is O(ni+nt) not O(ni*nt).
	ni := len(inter.mhdr)
	nt := int(x.mcount)
	xmhdr := (*[1 << 16]method)(add(unsafe.Pointer(x), uintptr(x.moff)))[:nt:nt]
	j := 0
	for k := 0; k < ni; k++ {
		i := &inter.mhdr[k]
		itype := inter.typ.typeOff(i.ityp)
		name := inter.typ.nameOff(i.name)
		iname := name.name()
		ipkg := name.pkgPath()
		if ipkg == "" {
			ipkg = inter.pkgpath.name()
		}
		for ; j < nt; j++ {
			t := &xmhdr[j]
			tname := typ.nameOff(t.name)
			if typ.typeOff(t.mtyp) == itype && tname.name() == iname {
				pkgPath := tname.pkgPath()
				if pkgPath == "" {
					pkgPath = typ.nameOff(x.pkgpath).name()
				}
				if tname.isExported() || pkgPath == ipkg {
					if m != nil {
						ifn := typ.textOff(t.ifn)
						*(*unsafe.Pointer)(add(unsafe.Pointer(&m.fun[0]), uintptr(k)*sys.PtrSize)) = ifn
					}
					goto nextimethod
				}
			}
		}
		// didn't find method
		if !canfail {
			if locked {
				unlock(&ifaceLock)
			}
			panic(&TypeAssertionError{"", typ.string(), inter.typ.string(), iname})
		}
		m.bad = true
		break
	nextimethod:
	}
	if !locked {
		throw("invalid itab locking")
	}
	h := itabhash(inter, typ)
	m.link = hash[h]
	m.inhash = true
	atomicstorep(unsafe.Pointer(&hash[h]), unsafe.Pointer(m))
}

func itabsinit() {
	lock(&ifaceLock)
	for _, md := range activeModules() {
		for _, i := range md.itablinks {
			// itablinks is a slice of pointers to the itabs used in this
			// module. A given itab may be used in more than one module
			// and thanks to the way global symbol resolution works, the
			// pointed-to itab may already have been inserted into the
			// global 'hash'.
			if !i.inhash {
				additab(i, true, false)
			}
		}
	}
	unlock(&ifaceLock)
}

// panicdottypeE is called when doing an e.(T) conversion and the conversion fails.
// have = the dynamic type we have.
// want = the static type we're trying to convert to.
// iface = the static type we're converting from.
func panicdottypeE(have, want, iface *_type) {
	haveString := ""
	if have != nil {
		haveString = have.string()
	}
	panic(&TypeAssertionError{iface.string(), haveString, want.string(), ""})
}

// panicdottypeI is called when doing an i.(T) conversion and the conversion fails.
// Same args as panicdottypeE, but "have" is the dynamic itab we have.
func panicdottypeI(have *itab, want, iface *_type) {
	var t *_type
	if have != nil {
		t = have._type
	}
	panicdottypeE(t, want, iface)
}

// panicnildottype is called when doing a i.(T) conversion and the interface i is nil.
// want = the static type we're trying to convert to.
func panicnildottype(want *_type) {
	panic(&TypeAssertionError{"", "", want.string(), ""})
	// TODO: Add the static type we're converting from as well.
	// It might generate a better error message.
	// Just to match other nil conversion errors, we don't for now.
}

// The conv and assert functions below do very similar things.
// The convXXX functions are guaranteed by the compiler to succeed.
// The assertXXX functions may fail (either panicking or returning false,
// depending on whether they are 1-result or 2-result).
// The convXXX functions succeed on a nil input, whereas the assertXXX
// functions fail on a nil input.

func convT2E(t *_type, elem unsafe.Pointer) (e eface) {
	if raceenabled {
		raceReadObjectPC(t, elem, getcallerpc(unsafe.Pointer(&t)), funcPC(convT2E))
	}
	if msanenabled {
		msanread(elem, t.size)
	}
	x := mallocgc(t.size, t, true)
	// TODO: We allocate a zeroed object only to overwrite it with actual data.
	// Figure out how to avoid zeroing. Also below in convT2Eslice, convT2I, convT2Islice.
	typedmemmove(t, x, elem)
	e._type = t
	e.data = x
	return
}

func convT2E16(t *_type, elem unsafe.Pointer) (e eface) {
	if raceenabled {
		raceReadObjectPC(t, elem, getcallerpc(unsafe.Pointer(&t)), funcPC(convT2E16))
	}
	if msanenabled {
		msanread(elem, t.size)
	}
	var x unsafe.Pointer
	if *(*uint16)(elem) == 0 {
		x = unsafe.Pointer(&zeroVal[0])
	} else {
		x = mallocgc(2, t, false)
		*(*uint16)(x) = *(*uint16)(elem)
	}
	e._type = t
	e.data = x
	return
}

func convT2E32(t *_type, elem unsafe.Pointer) (e eface) {
	if raceenabled {
		raceReadObjectPC(t, elem, getcallerpc(unsafe.Pointer(&t)), funcPC(convT2E32))
	}
	if msanenabled {
		msanread(elem, t.size)
	}
	var x unsafe.Pointer
	if *(*uint32)(elem) == 0 {
		x = unsafe.Pointer(&zeroVal[0])
	} else {
		x = mallocgc(4, t, false)
		*(*uint32)(x) = *(*uint32)(elem)
	}
	e._type = t
	e.data = x
	return
}

func convT2E64(t *_type, elem unsafe.Pointer) (e eface) {
	if raceenabled {
		raceReadObjectPC(t, elem, getcallerpc(unsafe.Pointer(&t)), funcPC(convT2E64))
	}
	if msanenabled {
		msanread(elem, t.size)
	}
	var x unsafe.Pointer
	if *(*uint64)(elem) == 0 {
		x = unsafe.Pointer(&zeroVal[0])
	} else {
		x = mallocgc(8, t, false)
		*(*uint64)(x) = *(*uint64)(elem)
	}
	e._type = t
	e.data = x
	return
}

func convT2Estring(t *_type, elem unsafe.Pointer) (e eface) {
	if raceenabled {
		raceReadObjectPC(t, elem, getcallerpc(unsafe.Pointer(&t)), funcPC(convT2Estring))
	}
	if msanenabled {
		msanread(elem, t.size)
	}
	var x unsafe.Pointer
	if *(*string)(elem) == "" {
		x = unsafe.Pointer(&zeroVal[0])
	} else {
		x = mallocgc(t.size, t, true)
		*(*string)(x) = *(*string)(elem)
	}
	e._type = t
	e.data = x
	return
}

func convT2Eslice(t *_type, elem unsafe.Pointer) (e eface) {
	if raceenabled {
		raceReadObjectPC(t, elem, getcallerpc(unsafe.Pointer(&t)), funcPC(convT2Eslice))
	}
	if msanenabled {
		msanread(elem, t.size)
	}
	var x unsafe.Pointer
	if v := *(*slice)(elem); uintptr(v.array) == 0 {
		x = unsafe.Pointer(&zeroVal[0])
	} else {
		x = mallocgc(t.size, t, true)
		*(*slice)(x) = *(*slice)(elem)
	}
	e._type = t
	e.data = x
	return
}

func convT2Enoptr(t *_type, elem unsafe.Pointer) (e eface) {
	if raceenabled {
		raceReadObjectPC(t, elem, getcallerpc(unsafe.Pointer(&t)), funcPC(convT2Enoptr))
	}
	if msanenabled {
		msanread(elem, t.size)
	}
	x := mallocgc(t.size, t, false)
	memmove(x, elem, t.size)
	e._type = t
	e.data = x
	return
}

func convT2I(tab *itab, elem unsafe.Pointer) (i iface) {
	t := tab._type
	if raceenabled {
		raceReadObjectPC(t, elem, getcallerpc(unsafe.Pointer(&tab)), funcPC(convT2I))
	}
	if msanenabled {
		msanread(elem, t.size)
	}
	x := mallocgc(t.size, t, true)
	typedmemmove(t, x, elem)
	i.tab = tab
	i.data = x
	return
}

func convT2I16(tab *itab, elem unsafe.Pointer) (i iface) {
	t := tab._type
	if raceenabled {
		raceReadObjectPC(t, elem, getcallerpc(unsafe.Pointer(&tab)), funcPC(convT2I16))
	}
	if msanenabled {
		msanread(elem, t.size)
	}
	var x unsafe.Pointer
	if *(*uint16)(elem) == 0 {
		x = unsafe.Pointer(&zeroVal[0])
	} else {
		x = mallocgc(2, t, false)
		*(*uint16)(x) = *(*uint16)(elem)
	}
	i.tab = tab
	i.data = x
	return
}

func convT2I32(tab *itab, elem unsafe.Pointer) (i iface) {
	t := tab._type
	if raceenabled {
		raceReadObjectPC(t, elem, getcallerpc(unsafe.Pointer(&tab)), funcPC(convT2I32))
	}
	if msanenabled {
		msanread(elem, t.size)
	}
	var x unsafe.Pointer
	if *(*uint32)(elem) == 0 {
		x = unsafe.Pointer(&zeroVal[0])
	} else {
		x = mallocgc(4, t, false)
		*(*uint32)(x) = *(*uint32)(elem)
	}
	i.tab = tab
	i.data = x
	return
}

func convT2I64(tab *itab, elem unsafe.Pointer) (i iface) {
	t := tab._type
	if raceenabled {
		raceReadObjectPC(t, elem, getcallerpc(unsafe.Pointer(&tab)), funcPC(convT2I64))
	}
	if msanenabled {
		msanread(elem, t.size)
	}
	var x unsafe.Pointer
	if *(*uint64)(elem) == 0 {
		x = unsafe.Pointer(&zeroVal[0])
	} else {
		x = mallocgc(8, t, false)
		*(*uint64)(x) = *(*uint64)(elem)
	}
	i.tab = tab
	i.data = x
	return
}

func convT2Istring(tab *itab, elem unsafe.Pointer) (i iface) {
	t := tab._type
	if raceenabled {
		raceReadObjectPC(t, elem, getcallerpc(unsafe.Pointer(&tab)), funcPC(convT2Istring))
	}
	if msanenabled {
		msanread(elem, t.size)
	}
	var x unsafe.Pointer
	if *(*string)(elem) == "" {
		x = unsafe.Pointer(&zeroVal[0])
	} else {
		x = mallocgc(t.size, t, true)
		*(*string)(x) = *(*string)(elem)
	}
	i.tab = tab
	i.data = x
	return
}

func convT2Islice(tab *itab, elem unsafe.Pointer) (i iface) {
	t := tab._type
	if raceenabled {
		raceReadObjectPC(t, elem, getcallerpc(unsafe.Pointer(&tab)), funcPC(convT2Islice))
	}
	if msanenabled {
		msanread(elem, t.size)
	}
	var x unsafe.Pointer
	if v := *(*slice)(elem); uintptr(v.array) == 0 {
		x = unsafe.Pointer(&zeroVal[0])
	} else {
		x = mallocgc(t.size, t, true)
		*(*slice)(x) = *(*slice)(elem)
	}
	i.tab = tab
	i.data = x
	return
}

func convT2Inoptr(tab *itab, elem unsafe.Pointer) (i iface) {
	t := tab._type
	if raceenabled {
		raceReadObjectPC(t, elem, getcallerpc(unsafe.Pointer(&tab)), funcPC(convT2Inoptr))
	}
	if msanenabled {
		msanread(elem, t.size)
	}
	x := mallocgc(t.size, t, false)
	memmove(x, elem, t.size)
	i.tab = tab
	i.data = x
	return
}

func convI2I(inter *interfacetype, i iface) (r iface) {
	tab := i.tab
	if tab == nil {
		return
	}
	if tab.inter == inter {
		r.tab = tab
		r.data = i.data
		return
	}
	r.tab = getitab(inter, tab._type, false)
	r.data = i.data
	return
}

func assertI2I(inter *interfacetype, i iface) (r iface) {
	tab := i.tab
	if tab == nil {
		// explicit conversions require non-nil interface value.
		panic(&TypeAssertionError{"", "", inter.typ.string(), ""})
	}
	if tab.inter == inter {
		r.tab = tab
		r.data = i.data
		return
	}
	r.tab = getitab(inter, tab._type, false)
	r.data = i.data
	return
}

func assertI2I2(inter *interfacetype, i iface) (r iface, b bool) {
	tab := i.tab
	if tab == nil {
		return
	}
	if tab.inter != inter {
		tab = getitab(inter, tab._type, true)
		if tab == nil {
			return
		}
	}
	r.tab = tab
	r.data = i.data
	b = true
	return
}

func assertE2I(inter *interfacetype, e eface) (r iface) {
	t := e._type
	if t == nil {
		// explicit conversions require non-nil interface value.
		panic(&TypeAssertionError{"", "", inter.typ.string(), ""})
	}
	r.tab = getitab(inter, t, false)
	r.data = e.data
	return
}

func assertE2I2(inter *interfacetype, e eface) (r iface, b bool) {
	t := e._type
	if t == nil {
		return
	}
	tab := getitab(inter, t, true)
	if tab == nil {
		return
	}
	r.tab = tab
	r.data = e.data
	b = true
	return
}

//go:linkname reflect_ifaceE2I reflect.ifaceE2I
func reflect_ifaceE2I(inter *interfacetype, e eface, dst *iface) {
	*dst = assertE2I(inter, e)
}

func iterate_itabs(fn func(*itab)) {
	for _, h := range &hash {
		for ; h != nil; h = h.link {
			fn(h)
		}
	}
}

// staticbytes is used to avoid convT2E for byte-sized values.
var staticbytes = [...]byte{
	0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
	0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
	0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
	0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
	0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27,
	0x28, 0x29, 0x2a, 0x2b, 0x2c, 0x2d, 0x2e, 0x2f,
	0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37,
	0x38, 0x39, 0x3a, 0x3b, 0x3c, 0x3d, 0x3e, 0x3f,
	0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47,
	0x48, 0x49, 0x4a, 0x4b, 0x4c, 0x4d, 0x4e, 0x4f,
	0x50, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57,
	0x58, 0x59, 0x5a, 0x5b, 0x5c, 0x5d, 0x5e, 0x5f,
	0x60, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67,
	0x68, 0x69, 0x6a, 0x6b, 0x6c, 0x6d, 0x6e, 0x6f,
	0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x76, 0x77,
	0x78, 0x79, 0x7a, 0x7b, 0x7c, 0x7d, 0x7e, 0x7f,
	0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
	0x88, 0x89, 0x8a, 0x8b, 0x8c, 0x8d, 0x8e, 0x8f,
	0x90, 0x91, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97,
	0x98, 0x99, 0x9a, 0x9b, 0x9c, 0x9d, 0x9e, 0x9f,
	0xa0, 0xa1, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
	0xa8, 0xa9, 0xaa, 0xab, 0xac, 0xad, 0xae, 0xaf,
	0xb0, 0xb1, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7,
	0xb8, 0xb9, 0xba, 0xbb, 0xbc, 0xbd, 0xbe, 0xbf,
	0xc0, 0xc1, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7,
	0xc8, 0xc9, 0xca, 0xcb, 0xcc, 0xcd, 0xce, 0xcf,
	0xd0, 0xd1, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7,
	0xd8, 0xd9, 0xda, 0xdb, 0xdc, 0xdd, 0xde, 0xdf,
	0xe0, 0xe1, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7,
	0xe8, 0xe9, 0xea, 0xeb, 0xec, 0xed, 0xee, 0xef,
	0xf0, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7,
	0xf8, 0xf9, 0xfa, 0xfb, 0xfc, 0xfd, 0xfe, 0xff,
}

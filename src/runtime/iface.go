// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"unsafe"
)

const (
	hashSize = 1009
)

var (
	ifaceLock mutex // lock for accessing hash
	hash      [hashSize]*itab
)

// fInterface is our standard non-empty interface.  We use it instead
// of interface{f()} in function prototypes because gofmt insists on
// putting lots of newlines in the otherwise concise interface{f()}.
type fInterface interface {
	f()
}

func getitab(inter *interfacetype, typ *_type, canfail bool) *itab {
	if len(inter.mhdr) == 0 {
		gothrow("internal error - misuse of itab")
	}

	// easy case
	x := typ.x
	if x == nil {
		if canfail {
			return nil
		}
		i := (*imethod)(add(unsafe.Pointer(inter), unsafe.Sizeof(interfacetype{})))
		panic(&TypeAssertionError{"", *typ._string, *inter.typ._string, *i.name})
	}

	// compiler has provided some good hash codes for us.
	h := inter.typ.hash
	h += 17 * typ.hash
	// TODO(rsc): h += 23 * x.mhash ?
	h %= hashSize

	// look twice - once without lock, once with.
	// common case will be no lock contention.
	var m *itab
	var locked int
	for locked = 0; locked < 2; locked++ {
		if locked != 0 {
			lock(&ifaceLock)
		}
		for m = (*itab)(atomicloadp(unsafe.Pointer(&hash[h]))); m != nil; m = m.link {
			if m.inter == inter && m._type == typ {
				if m.bad != 0 {
					m = nil
					if !canfail {
						// this can only happen if the conversion
						// was already done once using the , ok form
						// and we have a cached negative result.
						// the cached result doesn't record which
						// interface function was missing, so jump
						// down to the interface check, which will
						// do more work but give a better error.
						goto search
					}
				}
				if locked != 0 {
					unlock(&ifaceLock)
				}
				return m
			}
		}
	}

	m = (*itab)(persistentalloc(unsafe.Sizeof(itab{})+uintptr(len(inter.mhdr))*ptrSize, 0, &memstats.other_sys))
	m.inter = inter
	m._type = typ

search:
	// both inter and typ have method sorted by name,
	// and interface names are unique,
	// so can iterate over both in lock step;
	// the loop is O(ni+nt) not O(ni*nt).
	ni := len(inter.mhdr)
	nt := len(x.mhdr)
	j := 0
	for k := 0; k < ni; k++ {
		i := (*imethod)(add(unsafe.Pointer(inter), unsafe.Sizeof(interfacetype{})+uintptr(k)*unsafe.Sizeof(imethod{})))
		iname := i.name
		ipkgpath := i.pkgpath
		itype := i._type
		for ; j < nt; j++ {
			t := (*method)(add(unsafe.Pointer(x), unsafe.Sizeof(uncommontype{})+uintptr(j)*unsafe.Sizeof(method{})))
			if t.mtyp == itype && t.name == iname && t.pkgpath == ipkgpath {
				if m != nil {
					*(*unsafe.Pointer)(add(unsafe.Pointer(m), unsafe.Sizeof(itab{})+uintptr(k)*ptrSize)) = t.ifn
				}
				goto nextimethod
			}
		}
		// didn't find method
		if !canfail {
			if locked != 0 {
				unlock(&ifaceLock)
			}
			panic(&TypeAssertionError{"", *typ._string, *inter.typ._string, *iname})
		}
		m.bad = 1
		break
	nextimethod:
	}
	if locked == 0 {
		gothrow("invalid itab locking")
	}
	m.link = hash[h]
	atomicstorep(unsafe.Pointer(&hash[h]), unsafe.Pointer(m))
	unlock(&ifaceLock)
	if m.bad != 0 {
		return nil
	}
	return m
}

func typ2Itab(t *_type, inter *interfacetype, cache **itab) *itab {
	tab := getitab(inter, t, false)
	atomicstorep(unsafe.Pointer(cache), unsafe.Pointer(tab))
	return tab
}

func convT2E(t *_type, elem unsafe.Pointer) (e interface{}) {
	size := uintptr(t.size)
	ep := (*eface)(unsafe.Pointer(&e))
	if isDirectIface(t) {
		ep._type = t
		memmove(unsafe.Pointer(&ep.data), elem, size)
	} else {
		x := newobject(t)
		// TODO: We allocate a zeroed object only to overwrite it with
		// actual data.  Figure out how to avoid zeroing.  Also below in convT2I.
		memmove(x, elem, size)
		ep._type = t
		ep.data = x
	}
	return
}

func convT2I(t *_type, inter *interfacetype, cache **itab, elem unsafe.Pointer) (i fInterface) {
	tab := (*itab)(atomicloadp(unsafe.Pointer(cache)))
	if tab == nil {
		tab = getitab(inter, t, false)
		atomicstorep(unsafe.Pointer(cache), unsafe.Pointer(tab))
	}
	size := uintptr(t.size)
	pi := (*iface)(unsafe.Pointer(&i))
	if isDirectIface(t) {
		pi.tab = tab
		memmove(unsafe.Pointer(&pi.data), elem, size)
	} else {
		x := newobject(t)
		memmove(x, elem, size)
		pi.tab = tab
		pi.data = x
	}
	return
}

// TODO: give these routines a pointer to the result area instead of writing
// extra data in the outargs section.  Then we can get rid of go:nosplit.
//go:nosplit
func assertI2T(t *_type, i fInterface) (r struct{}) {
	ip := (*iface)(unsafe.Pointer(&i))
	tab := ip.tab
	if tab == nil {
		panic(&TypeAssertionError{"", "", *t._string, ""})
	}
	if tab._type != t {
		panic(&TypeAssertionError{*tab.inter.typ._string, *tab._type._string, *t._string, ""})
	}
	size := uintptr(t.size)
	if isDirectIface(t) {
		memmove(unsafe.Pointer(&r), unsafe.Pointer(&ip.data), size)
	} else {
		memmove(unsafe.Pointer(&r), ip.data, size)
	}
	return
}

//go:nosplit
func assertI2T2(t *_type, i fInterface) (r byte) {
	ip := (*iface)(unsafe.Pointer(&i))
	size := uintptr(t.size)
	ok := (*bool)(add(unsafe.Pointer(&r), size))
	tab := ip.tab
	if tab == nil || tab._type != t {
		*ok = false
		memclr(unsafe.Pointer(&r), size)
		return
	}
	*ok = true
	if isDirectIface(t) {
		memmove(unsafe.Pointer(&r), unsafe.Pointer(&ip.data), size)
	} else {
		memmove(unsafe.Pointer(&r), ip.data, size)
	}
	return
}

func assertI2TOK(t *_type, i fInterface) bool {
	ip := (*iface)(unsafe.Pointer(&i))
	tab := ip.tab
	return tab != nil && tab._type == t
}

//go:nosplit
func assertE2T(t *_type, e interface{}) (r struct{}) {
	ep := (*eface)(unsafe.Pointer(&e))
	if ep._type == nil {
		panic(&TypeAssertionError{"", "", *t._string, ""})
	}
	if ep._type != t {
		panic(&TypeAssertionError{"", *ep._type._string, *t._string, ""})
	}
	size := uintptr(t.size)
	if isDirectIface(t) {
		memmove(unsafe.Pointer(&r), unsafe.Pointer(&ep.data), size)
	} else {
		memmove(unsafe.Pointer(&r), ep.data, size)
	}
	return
}

//go:nosplit
func assertE2T2(t *_type, e interface{}) (r byte) {
	ep := (*eface)(unsafe.Pointer(&e))
	size := uintptr(t.size)
	ok := (*bool)(add(unsafe.Pointer(&r), size))
	if ep._type != t {
		*ok = false
		memclr(unsafe.Pointer(&r), size)
		return
	}
	*ok = true
	if isDirectIface(t) {
		memmove(unsafe.Pointer(&r), unsafe.Pointer(&ep.data), size)
	} else {
		memmove(unsafe.Pointer(&r), ep.data, size)
	}
	return
}

func assertE2TOK(t *_type, e interface{}) bool {
	ep := (*eface)(unsafe.Pointer(&e))
	return t == ep._type
}

func convI2E(i fInterface) (r interface{}) {
	ip := (*iface)(unsafe.Pointer(&i))
	tab := ip.tab
	if tab == nil {
		return
	}
	rp := (*eface)(unsafe.Pointer(&r))
	rp._type = tab._type
	rp.data = ip.data
	return
}

func assertI2E(inter *interfacetype, i fInterface) (r interface{}) {
	ip := (*iface)(unsafe.Pointer(&i))
	tab := ip.tab
	if tab == nil {
		// explicit conversions require non-nil interface value.
		panic(&TypeAssertionError{"", "", *inter.typ._string, ""})
	}
	rp := (*eface)(unsafe.Pointer(&r))
	rp._type = tab._type
	rp.data = ip.data
	return
}

func assertI2E2(inter *interfacetype, i fInterface) (r interface{}, ok bool) {
	ip := (*iface)(unsafe.Pointer(&i))
	tab := ip.tab
	if tab == nil {
		return
	}
	rp := (*eface)(unsafe.Pointer(&r))
	rp._type = tab._type
	rp.data = ip.data
	ok = true
	return
}

func convI2I(inter *interfacetype, i fInterface) (r fInterface) {
	ip := (*iface)(unsafe.Pointer(&i))
	tab := ip.tab
	if tab == nil {
		return
	}
	rp := (*iface)(unsafe.Pointer(&r))
	if tab.inter == inter {
		rp.tab = tab
		rp.data = ip.data
		return
	}
	rp.tab = getitab(inter, tab._type, false)
	rp.data = ip.data
	return
}

func assertI2I(inter *interfacetype, i fInterface) (r fInterface) {
	ip := (*iface)(unsafe.Pointer(&i))
	tab := ip.tab
	if tab == nil {
		// explicit conversions require non-nil interface value.
		panic(&TypeAssertionError{"", "", *inter.typ._string, ""})
	}
	rp := (*iface)(unsafe.Pointer(&r))
	if tab.inter == inter {
		rp.tab = tab
		rp.data = ip.data
		return
	}
	rp.tab = getitab(inter, tab._type, false)
	rp.data = ip.data
	return
}

func assertI2I2(inter *interfacetype, i fInterface) (r fInterface, ok bool) {
	ip := (*iface)(unsafe.Pointer(&i))
	tab := ip.tab
	if tab == nil {
		return
	}
	rp := (*iface)(unsafe.Pointer(&r))
	if tab.inter == inter {
		rp.tab = tab
		rp.data = ip.data
		ok = true
		return
	}
	tab = getitab(inter, tab._type, true)
	if tab == nil {
		rp.data = nil
		rp.tab = nil
		ok = false
		return
	}
	rp.tab = tab
	rp.data = ip.data
	ok = true
	return
}

func assertE2I(inter *interfacetype, e interface{}) (r fInterface) {
	ep := (*eface)(unsafe.Pointer(&e))
	t := ep._type
	if t == nil {
		// explicit conversions require non-nil interface value.
		panic(&TypeAssertionError{"", "", *inter.typ._string, ""})
	}
	rp := (*iface)(unsafe.Pointer(&r))
	rp.tab = getitab(inter, t, false)
	rp.data = ep.data
	return
}

func assertE2I2(inter *interfacetype, e interface{}) (r fInterface, ok bool) {
	ep := (*eface)(unsafe.Pointer(&e))
	t := ep._type
	if t == nil {
		return
	}
	tab := getitab(inter, t, true)
	if tab == nil {
		return
	}
	rp := (*iface)(unsafe.Pointer(&r))
	rp.tab = tab
	rp.data = ep.data
	ok = true
	return
}

//go:linkname reflect_ifaceE2I reflect.ifaceE2I
func reflect_ifaceE2I(inter *interfacetype, e interface{}, dst *fInterface) {
	*dst = assertE2I(inter, e)
}

func assertE2E(inter *interfacetype, e interface{}) interface{} {
	ep := (*eface)(unsafe.Pointer(&e))
	if ep._type == nil {
		// explicit conversions require non-nil interface value.
		panic(&TypeAssertionError{"", "", *inter.typ._string, ""})
	}
	return e
}

func assertE2E2(inter *interfacetype, e interface{}) (interface{}, bool) {
	ep := (*eface)(unsafe.Pointer(&e))
	if ep._type == nil {
		return nil, false
	}
	return e, true
}

func ifacethash(i fInterface) uint32 {
	ip := (*iface)(unsafe.Pointer(&i))
	tab := ip.tab
	if tab == nil {
		return 0
	}
	return tab._type.hash
}

func efacethash(e interface{}) uint32 {
	ep := (*eface)(unsafe.Pointer(&e))
	t := ep._type
	if t == nil {
		return 0
	}
	return t.hash
}

func iterate_itabs(fn func(*itab)) {
	for _, h := range &hash {
		for ; h != nil; h = h.link {
			fn(h)
		}
	}
}

func ifaceE2I2(inter *interfacetype, e interface{}, r *fInterface) (ok bool) {
	*r, ok = assertE2I2(inter, e)
	return
}

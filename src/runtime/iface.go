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
		throw("internal error - misuse of itab")
	}

	// easy case
	x := typ.x
	if x == nil {
		if canfail {
			return nil
		}
		panic(&TypeAssertionError{"", *typ._string, *inter.typ._string, *inter.mhdr[0].name})
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

	m = (*itab)(persistentalloc(unsafe.Sizeof(itab{})+uintptr(len(inter.mhdr)-1)*ptrSize, 0, &memstats.other_sys))
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
		i := &inter.mhdr[k]
		iname := i.name
		ipkgpath := i.pkgpath
		itype := i._type
		for ; j < nt; j++ {
			t := &x.mhdr[j]
			if t.mtyp == itype && t.name == iname && t.pkgpath == ipkgpath {
				if m != nil {
					*(*unsafe.Pointer)(add(unsafe.Pointer(&m.fun[0]), uintptr(k)*ptrSize)) = t.ifn
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
		throw("invalid itab locking")
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
	ep := (*eface)(unsafe.Pointer(&e))
	if isDirectIface(t) {
		ep._type = t
		typedmemmove(t, unsafe.Pointer(&ep.data), elem)
	} else {
		x := newobject(t)
		// TODO: We allocate a zeroed object only to overwrite it with
		// actual data.  Figure out how to avoid zeroing.  Also below in convT2I.
		typedmemmove(t, x, elem)
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
	pi := (*iface)(unsafe.Pointer(&i))
	if isDirectIface(t) {
		pi.tab = tab
		typedmemmove(t, unsafe.Pointer(&pi.data), elem)
	} else {
		x := newobject(t)
		typedmemmove(t, x, elem)
		pi.tab = tab
		pi.data = x
	}
	return
}

func assertI2T(t *_type, i fInterface, r unsafe.Pointer) {
	ip := (*iface)(unsafe.Pointer(&i))
	tab := ip.tab
	if tab == nil {
		panic(&TypeAssertionError{"", "", *t._string, ""})
	}
	if tab._type != t {
		panic(&TypeAssertionError{*tab.inter.typ._string, *tab._type._string, *t._string, ""})
	}
	if r != nil {
		if isDirectIface(t) {
			writebarrierptr((*uintptr)(r), uintptr(ip.data))
		} else {
			typedmemmove(t, r, ip.data)
		}
	}
}

func assertI2T2(t *_type, i fInterface, r unsafe.Pointer) bool {
	ip := (*iface)(unsafe.Pointer(&i))
	tab := ip.tab
	if tab == nil || tab._type != t {
		if r != nil {
			memclr(r, uintptr(t.size))
		}
		return false
	}
	if r != nil {
		if isDirectIface(t) {
			writebarrierptr((*uintptr)(r), uintptr(ip.data))
		} else {
			typedmemmove(t, r, ip.data)
		}
	}
	return true
}

func assertE2T(t *_type, e interface{}, r unsafe.Pointer) {
	ep := (*eface)(unsafe.Pointer(&e))
	if ep._type == nil {
		panic(&TypeAssertionError{"", "", *t._string, ""})
	}
	if ep._type != t {
		panic(&TypeAssertionError{"", *ep._type._string, *t._string, ""})
	}
	if r != nil {
		if isDirectIface(t) {
			writebarrierptr((*uintptr)(r), uintptr(ep.data))
		} else {
			typedmemmove(t, r, ep.data)
		}
	}
}

func assertE2T2(t *_type, e interface{}, r unsafe.Pointer) bool {
	ep := (*eface)(unsafe.Pointer(&e))
	if ep._type != t {
		if r != nil {
			memclr(r, uintptr(t.size))
		}
		return false
	}
	if r != nil {
		if isDirectIface(t) {
			writebarrierptr((*uintptr)(r), uintptr(ep.data))
		} else {
			typedmemmove(t, r, ep.data)
		}
	}
	return true
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

func assertI2E(inter *interfacetype, i fInterface, r *interface{}) {
	ip := (*iface)(unsafe.Pointer(&i))
	tab := ip.tab
	if tab == nil {
		// explicit conversions require non-nil interface value.
		panic(&TypeAssertionError{"", "", *inter.typ._string, ""})
	}
	rp := (*eface)(unsafe.Pointer(r))
	rp._type = tab._type
	rp.data = ip.data
	return
}

func assertI2E2(inter *interfacetype, i fInterface, r *interface{}) bool {
	ip := (*iface)(unsafe.Pointer(&i))
	tab := ip.tab
	if tab == nil {
		return false
	}
	if r != nil {
		rp := (*eface)(unsafe.Pointer(r))
		rp._type = tab._type
		rp.data = ip.data
	}
	return true
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

func assertI2I(inter *interfacetype, i fInterface, r *fInterface) {
	ip := (*iface)(unsafe.Pointer(&i))
	tab := ip.tab
	if tab == nil {
		// explicit conversions require non-nil interface value.
		panic(&TypeAssertionError{"", "", *inter.typ._string, ""})
	}
	rp := (*iface)(unsafe.Pointer(r))
	if tab.inter == inter {
		rp.tab = tab
		rp.data = ip.data
		return
	}
	rp.tab = getitab(inter, tab._type, false)
	rp.data = ip.data
}

func assertI2I2(inter *interfacetype, i fInterface, r *fInterface) bool {
	ip := (*iface)(unsafe.Pointer(&i))
	tab := ip.tab
	if tab == nil {
		if r != nil {
			*r = nil
		}
		return false
	}
	if tab.inter != inter {
		tab = getitab(inter, tab._type, true)
		if tab == nil {
			if r != nil {
				*r = nil
			}
			return false
		}
	}
	if r != nil {
		rp := (*iface)(unsafe.Pointer(r))
		rp.tab = tab
		rp.data = ip.data
	}
	return true
}

func assertE2I(inter *interfacetype, e interface{}, r *fInterface) {
	ep := (*eface)(unsafe.Pointer(&e))
	t := ep._type
	if t == nil {
		// explicit conversions require non-nil interface value.
		panic(&TypeAssertionError{"", "", *inter.typ._string, ""})
	}
	rp := (*iface)(unsafe.Pointer(r))
	rp.tab = getitab(inter, t, false)
	rp.data = ep.data
}

func assertE2I2(inter *interfacetype, e interface{}, r *fInterface) bool {
	ep := (*eface)(unsafe.Pointer(&e))
	t := ep._type
	if t == nil {
		if r != nil {
			*r = nil
		}
		return false
	}
	tab := getitab(inter, t, true)
	if tab == nil {
		if r != nil {
			*r = nil
		}
		return false
	}
	if r != nil {
		rp := (*iface)(unsafe.Pointer(r))
		rp.tab = tab
		rp.data = ep.data
	}
	return true
}

//go:linkname reflect_ifaceE2I reflect.ifaceE2I
func reflect_ifaceE2I(inter *interfacetype, e interface{}, dst *fInterface) {
	assertE2I(inter, e, dst)
}

func assertE2E(inter *interfacetype, e interface{}, r *interface{}) {
	ep := (*eface)(unsafe.Pointer(&e))
	if ep._type == nil {
		// explicit conversions require non-nil interface value.
		panic(&TypeAssertionError{"", "", *inter.typ._string, ""})
	}
	*r = e
}

func assertE2E2(inter *interfacetype, e interface{}, r *interface{}) bool {
	ep := (*eface)(unsafe.Pointer(&e))
	if ep._type == nil {
		if r != nil {
			*r = nil
		}
		return false
	}
	if r != nil {
		*r = e
	}
	return true
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

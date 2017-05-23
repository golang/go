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
				if m.bad != 0 {
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
	if m.bad != 0 {
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
		m.bad = 1
		break
	nextimethod:
	}
	if !locked {
		throw("invalid itab locking")
	}
	h := itabhash(inter, typ)
	m.link = hash[h]
	atomicstorep(unsafe.Pointer(&hash[h]), unsafe.Pointer(m))
}

func itabsinit() {
	lock(&ifaceLock)
	for m := &firstmoduledata; m != nil; m = m.next {
		for _, i := range m.itablinks {
			additab(i, true, false)
		}
	}
	unlock(&ifaceLock)
}

func convT2E(t *_type, elem unsafe.Pointer, x unsafe.Pointer) (e eface) {
	if raceenabled {
		raceReadObjectPC(t, elem, getcallerpc(unsafe.Pointer(&t)), funcPC(convT2E))
	}
	if msanenabled {
		msanread(elem, t.size)
	}
	if isDirectIface(t) {
		throw("direct convT2E")
	}
	if x == nil {
		x = newobject(t)
		// TODO: We allocate a zeroed object only to overwrite it with
		// actual data. Figure out how to avoid zeroing. Also below in convT2I.
	}
	typedmemmove(t, x, elem)
	e._type = t
	e.data = x
	return
}

func convT2I(tab *itab, elem unsafe.Pointer, x unsafe.Pointer) (i iface) {
	t := tab._type
	if raceenabled {
		raceReadObjectPC(t, elem, getcallerpc(unsafe.Pointer(&tab)), funcPC(convT2I))
	}
	if msanenabled {
		msanread(elem, t.size)
	}
	if isDirectIface(t) {
		throw("direct convT2I")
	}
	if x == nil {
		x = newobject(t)
	}
	typedmemmove(t, x, elem)
	i.tab = tab
	i.data = x
	return
}

func panicdottype(have, want, iface *_type) {
	haveString := ""
	if have != nil {
		haveString = have.string()
	}
	panic(&TypeAssertionError{iface.string(), haveString, want.string(), ""})
}

func assertI2T(t *_type, i iface, r unsafe.Pointer) {
	tab := i.tab
	if tab == nil {
		panic(&TypeAssertionError{"", "", t.string(), ""})
	}
	if tab._type != t {
		panic(&TypeAssertionError{tab.inter.typ.string(), tab._type.string(), t.string(), ""})
	}
	if r != nil {
		if isDirectIface(t) {
			writebarrierptr((*uintptr)(r), uintptr(i.data))
		} else {
			typedmemmove(t, r, i.data)
		}
	}
}

func assertI2T2(t *_type, i iface, r unsafe.Pointer) bool {
	tab := i.tab
	if tab == nil || tab._type != t {
		if r != nil {
			memclr(r, t.size)
		}
		return false
	}
	if r != nil {
		if isDirectIface(t) {
			writebarrierptr((*uintptr)(r), uintptr(i.data))
		} else {
			typedmemmove(t, r, i.data)
		}
	}
	return true
}

func assertE2T(t *_type, e eface, r unsafe.Pointer) {
	if e._type == nil {
		panic(&TypeAssertionError{"", "", t.string(), ""})
	}
	if e._type != t {
		panic(&TypeAssertionError{"", e._type.string(), t.string(), ""})
	}
	if r != nil {
		if isDirectIface(t) {
			writebarrierptr((*uintptr)(r), uintptr(e.data))
		} else {
			typedmemmove(t, r, e.data)
		}
	}
}

var testingAssertE2T2GC bool

// The compiler ensures that r is non-nil.
func assertE2T2(t *_type, e eface, r unsafe.Pointer) bool {
	if testingAssertE2T2GC {
		GC()
	}
	if e._type != t {
		memclr(r, t.size)
		return false
	}
	if isDirectIface(t) {
		writebarrierptr((*uintptr)(r), uintptr(e.data))
	} else {
		typedmemmove(t, r, e.data)
	}
	return true
}

func convI2E(i iface) (r eface) {
	tab := i.tab
	if tab == nil {
		return
	}
	r._type = tab._type
	r.data = i.data
	return
}

func assertI2E(inter *interfacetype, i iface, r *eface) {
	tab := i.tab
	if tab == nil {
		// explicit conversions require non-nil interface value.
		panic(&TypeAssertionError{"", "", inter.typ.string(), ""})
	}
	r._type = tab._type
	r.data = i.data
	return
}

// The compiler ensures that r is non-nil.
func assertI2E2(inter *interfacetype, i iface, r *eface) bool {
	tab := i.tab
	if tab == nil {
		return false
	}
	r._type = tab._type
	r.data = i.data
	return true
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

func assertI2I(inter *interfacetype, i iface, r *iface) {
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
}

func assertI2I2(inter *interfacetype, i iface, r *iface) bool {
	tab := i.tab
	if tab == nil {
		if r != nil {
			*r = iface{}
		}
		return false
	}
	if tab.inter != inter {
		tab = getitab(inter, tab._type, true)
		if tab == nil {
			if r != nil {
				*r = iface{}
			}
			return false
		}
	}
	if r != nil {
		r.tab = tab
		r.data = i.data
	}
	return true
}

func assertE2I(inter *interfacetype, e eface, r *iface) {
	t := e._type
	if t == nil {
		// explicit conversions require non-nil interface value.
		panic(&TypeAssertionError{"", "", inter.typ.string(), ""})
	}
	r.tab = getitab(inter, t, false)
	r.data = e.data
}

var testingAssertE2I2GC bool

func assertE2I2(inter *interfacetype, e eface, r *iface) bool {
	if testingAssertE2I2GC {
		GC()
	}
	t := e._type
	if t == nil {
		if r != nil {
			*r = iface{}
		}
		return false
	}
	tab := getitab(inter, t, true)
	if tab == nil {
		if r != nil {
			*r = iface{}
		}
		return false
	}
	if r != nil {
		r.tab = tab
		r.data = e.data
	}
	return true
}

//go:linkname reflect_ifaceE2I reflect.ifaceE2I
func reflect_ifaceE2I(inter *interfacetype, e eface, dst *iface) {
	assertE2I(inter, e, dst)
}

func assertE2E(inter *interfacetype, e eface, r *eface) {
	if e._type == nil {
		// explicit conversions require non-nil interface value.
		panic(&TypeAssertionError{"", "", inter.typ.string(), ""})
	}
	*r = e
}

// The compiler ensures that r is non-nil.
func assertE2E2(inter *interfacetype, e eface, r *eface) bool {
	if e._type == nil {
		*r = eface{}
		return false
	}
	*r = e
	return true
}

func iterate_itabs(fn func(*itab)) {
	for _, h := range &hash {
		for ; h != nil; h = h.link {
			fn(h)
		}
	}
}

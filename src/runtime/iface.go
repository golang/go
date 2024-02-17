// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/abi"
	"internal/goarch"
	"runtime/internal/atomic"
	"runtime/internal/sys"
	"unsafe"
)

const itabInitSize = 512

var (
	itabLock      mutex                               // lock for accessing itab table
	itabTable     = &itabTableInit                    // pointer to current table
	itabTableInit = itabTableType{size: itabInitSize} // starter table
)

// Note: change the formula in the mallocgc call in itabAdd if you change these fields.
type itabTableType struct {
	size    uintptr             // length of entries array. Always a power of 2.
	count   uintptr             // current number of filled entries.
	entries [itabInitSize]*itab // really [size] large
}

func itabHashFunc(inter *interfacetype, typ *_type) uintptr {
	// compiler has provided some good hash codes for us.
	return uintptr(inter.Type.Hash ^ typ.Hash)
}

func getitab(inter *interfacetype, typ *_type, canfail bool) *itab {
	if len(inter.Methods) == 0 {
		throw("internal error - misuse of itab")
	}

	// easy case
	if typ.TFlag&abi.TFlagUncommon == 0 {
		if canfail {
			return nil
		}
		name := toRType(&inter.Type).nameOff(inter.Methods[0].Name)
		panic(&TypeAssertionError{nil, typ, &inter.Type, name.Name()})
	}

	var m *itab

	// First, look in the existing table to see if we can find the itab we need.
	// This is by far the most common case, so do it without locks.
	// Use atomic to ensure we see any previous writes done by the thread
	// that updates the itabTable field (with atomic.Storep in itabAdd).
	t := (*itabTableType)(atomic.Loadp(unsafe.Pointer(&itabTable)))
	if m = t.find(inter, typ); m != nil {
		goto finish
	}

	// Not found.  Grab the lock and try again.
	lock(&itabLock)
	if m = itabTable.find(inter, typ); m != nil {
		unlock(&itabLock)
		goto finish
	}

	// Entry doesn't exist yet. Make a new entry & add it.
	m = (*itab)(persistentalloc(unsafe.Sizeof(itab{})+uintptr(len(inter.Methods)-1)*goarch.PtrSize, 0, &memstats.other_sys))
	m.Inter = inter
	m.Type = typ
	// The hash is used in type switches. However, compiler statically generates itab's
	// for all interface/type pairs used in switches (which are added to itabTable
	// in itabsinit). The dynamically-generated itab's never participate in type switches,
	// and thus the hash is irrelevant.
	// Note: m.Hash is _not_ the hash used for the runtime itabTable hash table.
	m.Hash = 0
	itabInit(m)
	itabAdd(m)
	unlock(&itabLock)
finish:
	if m.Fun[0] != 0 {
		return m
	}
	if canfail {
		return nil
	}
	// this can only happen if the conversion
	// was already done once using the , ok form
	// and we have a cached negative result.
	// The cached result doesn't record which
	// interface function was missing, so initialize
	// the itab again to get the missing function name.
	panic(&TypeAssertionError{concrete: typ, asserted: &inter.Type, missingMethod: itabInit(m)})
}

// find finds the given interface/type pair in t.
// Returns nil if the given interface/type pair isn't present.
func (t *itabTableType) find(inter *interfacetype, typ *_type) *itab {
	// Implemented using quadratic probing.
	// Probe sequence is h(i) = h0 + i*(i+1)/2 mod 2^k.
	// We're guaranteed to hit all table entries using this probe sequence.
	mask := t.size - 1
	h := itabHashFunc(inter, typ) & mask
	for i := uintptr(1); ; i++ {
		p := (**itab)(add(unsafe.Pointer(&t.entries), h*goarch.PtrSize))
		// Use atomic read here so if we see m != nil, we also see
		// the initializations of the fields of m.
		// m := *p
		m := (*itab)(atomic.Loadp(unsafe.Pointer(p)))
		if m == nil {
			return nil
		}
		if m.Inter == inter && m.Type == typ {
			return m
		}
		h += i
		h &= mask
	}
}

// itabAdd adds the given itab to the itab hash table.
// itabLock must be held.
func itabAdd(m *itab) {
	// Bugs can lead to calling this while mallocing is set,
	// typically because this is called while panicking.
	// Crash reliably, rather than only when we need to grow
	// the hash table.
	if getg().m.mallocing != 0 {
		throw("malloc deadlock")
	}

	t := itabTable
	if t.count >= 3*(t.size/4) { // 75% load factor
		// Grow hash table.
		// t2 = new(itabTableType) + some additional entries
		// We lie and tell malloc we want pointer-free memory because
		// all the pointed-to values are not in the heap.
		t2 := (*itabTableType)(mallocgc((2+2*t.size)*goarch.PtrSize, nil, true))
		t2.size = t.size * 2

		// Copy over entries.
		// Note: while copying, other threads may look for an itab and
		// fail to find it. That's ok, they will then try to get the itab lock
		// and as a consequence wait until this copying is complete.
		iterate_itabs(t2.add)
		if t2.count != t.count {
			throw("mismatched count during itab table copy")
		}
		// Publish new hash table. Use an atomic write: see comment in getitab.
		atomicstorep(unsafe.Pointer(&itabTable), unsafe.Pointer(t2))
		// Adopt the new table as our own.
		t = itabTable
		// Note: the old table can be GC'ed here.
	}
	t.add(m)
}

// add adds the given itab to itab table t.
// itabLock must be held.
func (t *itabTableType) add(m *itab) {
	// See comment in find about the probe sequence.
	// Insert new itab in the first empty spot in the probe sequence.
	mask := t.size - 1
	h := itabHashFunc(m.Inter, m.Type) & mask
	for i := uintptr(1); ; i++ {
		p := (**itab)(add(unsafe.Pointer(&t.entries), h*goarch.PtrSize))
		m2 := *p
		if m2 == m {
			// A given itab may be used in more than one module
			// and thanks to the way global symbol resolution works, the
			// pointed-to itab may already have been inserted into the
			// global 'hash'.
			return
		}
		if m2 == nil {
			// Use atomic write here so if a reader sees m, it also
			// sees the correctly initialized fields of m.
			// NoWB is ok because m is not in heap memory.
			// *p = m
			atomic.StorepNoWB(unsafe.Pointer(p), unsafe.Pointer(m))
			t.count++
			return
		}
		h += i
		h &= mask
	}
}

// init fills in the m.Fun array with all the code pointers for
// the m.Inter/m.Type pair. If the type does not implement the interface,
// it sets m.Fun[0] to 0 and returns the name of an interface function that is missing.
// It is ok to call this multiple times on the same m, even concurrently.
func itabInit(m *itab) string {
	inter := m.Inter
	typ := m.Type
	x := typ.Uncommon()

	// both inter and typ have method sorted by name,
	// and interface names are unique,
	// so can iterate over both in lock step;
	// the loop is O(ni+nt) not O(ni*nt).
	ni := len(inter.Methods)
	nt := int(x.Mcount)
	xmhdr := (*[1 << 16]abi.Method)(add(unsafe.Pointer(x), uintptr(x.Moff)))[:nt:nt]
	j := 0
	methods := (*[1 << 16]unsafe.Pointer)(unsafe.Pointer(&m.Fun[0]))[:ni:ni]
	var fun0 unsafe.Pointer
imethods:
	for k := 0; k < ni; k++ {
		i := &inter.Methods[k]
		itype := toRType(&inter.Type).typeOff(i.Typ)
		name := toRType(&inter.Type).nameOff(i.Name)
		iname := name.Name()
		ipkg := pkgPath(name)
		if ipkg == "" {
			ipkg = inter.PkgPath.Name()
		}
		for ; j < nt; j++ {
			t := &xmhdr[j]
			rtyp := toRType(typ)
			tname := rtyp.nameOff(t.Name)
			if rtyp.typeOff(t.Mtyp) == itype && tname.Name() == iname {
				pkgPath := pkgPath(tname)
				if pkgPath == "" {
					pkgPath = rtyp.nameOff(x.PkgPath).Name()
				}
				if tname.IsExported() || pkgPath == ipkg {
					ifn := rtyp.textOff(t.Ifn)
					if k == 0 {
						fun0 = ifn // we'll set m.Fun[0] at the end
					} else {
						methods[k] = ifn
					}
					continue imethods
				}
			}
		}
		// didn't find method
		m.Fun[0] = 0
		return iname
	}
	m.Fun[0] = uintptr(fun0)
	return ""
}

func itabsinit() {
	lockInit(&itabLock, lockRankItab)
	lock(&itabLock)
	for _, md := range activeModules() {
		for _, i := range md.itablinks {
			itabAdd(i)
		}
	}
	unlock(&itabLock)
}

// panicdottypeE is called when doing an e.(T) conversion and the conversion fails.
// have = the dynamic type we have.
// want = the static type we're trying to convert to.
// iface = the static type we're converting from.
func panicdottypeE(have, want, iface *_type) {
	panic(&TypeAssertionError{iface, have, want, ""})
}

// panicdottypeI is called when doing an i.(T) conversion and the conversion fails.
// Same args as panicdottypeE, but "have" is the dynamic itab we have.
func panicdottypeI(have *itab, want, iface *_type) {
	var t *_type
	if have != nil {
		t = have.Type
	}
	panicdottypeE(t, want, iface)
}

// panicnildottype is called when doing an i.(T) conversion and the interface i is nil.
// want = the static type we're trying to convert to.
func panicnildottype(want *_type) {
	panic(&TypeAssertionError{nil, nil, want, ""})
	// TODO: Add the static type we're converting from as well.
	// It might generate a better error message.
	// Just to match other nil conversion errors, we don't for now.
}

// The specialized convTx routines need a type descriptor to use when calling mallocgc.
// We don't need the type to be exact, just to have the correct size, alignment, and pointer-ness.
// However, when debugging, it'd be nice to have some indication in mallocgc where the types came from,
// so we use named types here.
// We then construct interface values of these types,
// and then extract the type word to use as needed.
type (
	uint16InterfacePtr uint16
	uint32InterfacePtr uint32
	uint64InterfacePtr uint64
	stringInterfacePtr string
	sliceInterfacePtr  []byte
)

var (
	uint16Eface any = uint16InterfacePtr(0)
	uint32Eface any = uint32InterfacePtr(0)
	uint64Eface any = uint64InterfacePtr(0)
	stringEface any = stringInterfacePtr("")
	sliceEface  any = sliceInterfacePtr(nil)

	uint16Type *_type = efaceOf(&uint16Eface)._type
	uint32Type *_type = efaceOf(&uint32Eface)._type
	uint64Type *_type = efaceOf(&uint64Eface)._type
	stringType *_type = efaceOf(&stringEface)._type
	sliceType  *_type = efaceOf(&sliceEface)._type
)

// The conv and assert functions below do very similar things.
// The convXXX functions are guaranteed by the compiler to succeed.
// The assertXXX functions may fail (either panicking or returning false,
// depending on whether they are 1-result or 2-result).
// The convXXX functions succeed on a nil input, whereas the assertXXX
// functions fail on a nil input.

// convT converts a value of type t, which is pointed to by v, to a pointer that can
// be used as the second word of an interface value.
func convT(t *_type, v unsafe.Pointer) unsafe.Pointer {
	if raceenabled {
		raceReadObjectPC(t, v, getcallerpc(), abi.FuncPCABIInternal(convT))
	}
	if msanenabled {
		msanread(v, t.Size_)
	}
	if asanenabled {
		asanread(v, t.Size_)
	}
	x := mallocgc(t.Size_, t, true)
	typedmemmove(t, x, v)
	return x
}
func convTnoptr(t *_type, v unsafe.Pointer) unsafe.Pointer {
	// TODO: maybe take size instead of type?
	if raceenabled {
		raceReadObjectPC(t, v, getcallerpc(), abi.FuncPCABIInternal(convTnoptr))
	}
	if msanenabled {
		msanread(v, t.Size_)
	}
	if asanenabled {
		asanread(v, t.Size_)
	}

	x := mallocgc(t.Size_, t, false)
	memmove(x, v, t.Size_)
	return x
}

func convT16(val uint16) (x unsafe.Pointer) {
	if val < uint16(len(staticuint64s)) {
		x = unsafe.Pointer(&staticuint64s[val])
		if goarch.BigEndian {
			x = add(x, 6)
		}
	} else {
		x = mallocgc(2, uint16Type, false)
		*(*uint16)(x) = val
	}
	return
}

func convT32(val uint32) (x unsafe.Pointer) {
	if val < uint32(len(staticuint64s)) {
		x = unsafe.Pointer(&staticuint64s[val])
		if goarch.BigEndian {
			x = add(x, 4)
		}
	} else {
		x = mallocgc(4, uint32Type, false)
		*(*uint32)(x) = val
	}
	return
}

func convT64(val uint64) (x unsafe.Pointer) {
	if val < uint64(len(staticuint64s)) {
		x = unsafe.Pointer(&staticuint64s[val])
	} else {
		x = mallocgc(8, uint64Type, false)
		*(*uint64)(x) = val
	}
	return
}

func convTstring(val string) (x unsafe.Pointer) {
	if val == "" {
		x = unsafe.Pointer(&zeroVal[0])
	} else {
		x = mallocgc(unsafe.Sizeof(val), stringType, true)
		*(*string)(x) = val
	}
	return
}

func convTslice(val []byte) (x unsafe.Pointer) {
	// Note: this must work for any element type, not just byte.
	if (*slice)(unsafe.Pointer(&val)).array == nil {
		x = unsafe.Pointer(&zeroVal[0])
	} else {
		x = mallocgc(unsafe.Sizeof(val), sliceType, true)
		*(*[]byte)(x) = val
	}
	return
}

func assertE2I(inter *interfacetype, t *_type) *itab {
	if t == nil {
		// explicit conversions require non-nil interface value.
		panic(&TypeAssertionError{nil, nil, &inter.Type, ""})
	}
	return getitab(inter, t, false)
}

func assertE2I2(inter *interfacetype, t *_type) *itab {
	if t == nil {
		return nil
	}
	return getitab(inter, t, true)
}

// typeAssert builds an itab for the concrete type t and the
// interface type s.Inter. If the conversion is not possible it
// panics if s.CanFail is false and returns nil if s.CanFail is true.
func typeAssert(s *abi.TypeAssert, t *_type) *itab {
	var tab *itab
	if t == nil {
		if !s.CanFail {
			panic(&TypeAssertionError{nil, nil, &s.Inter.Type, ""})
		}
	} else {
		tab = getitab(s.Inter, t, s.CanFail)
	}

	if !abi.UseInterfaceSwitchCache(GOARCH) {
		return tab
	}

	// Maybe update the cache, so the next time the generated code
	// doesn't need to call into the runtime.
	if cheaprand()&1023 != 0 {
		// Only bother updating the cache ~1 in 1000 times.
		return tab
	}
	// Load the current cache.
	oldC := (*abi.TypeAssertCache)(atomic.Loadp(unsafe.Pointer(&s.Cache)))

	if cheaprand()&uint32(oldC.Mask) != 0 {
		// As cache gets larger, choose to update it less often
		// so we can amortize the cost of building a new cache.
		return tab
	}

	// Make a new cache.
	newC := buildTypeAssertCache(oldC, t, tab)

	// Update cache. Use compare-and-swap so if multiple threads
	// are fighting to update the cache, at least one of their
	// updates will stick.
	atomic_casPointer((*unsafe.Pointer)(unsafe.Pointer(&s.Cache)), unsafe.Pointer(oldC), unsafe.Pointer(newC))

	return tab
}

func buildTypeAssertCache(oldC *abi.TypeAssertCache, typ *_type, tab *itab) *abi.TypeAssertCache {
	oldEntries := unsafe.Slice(&oldC.Entries[0], oldC.Mask+1)

	// Count the number of entries we need.
	n := 1
	for _, e := range oldEntries {
		if e.Typ != 0 {
			n++
		}
	}

	// Figure out how big a table we need.
	// We need at least one more slot than the number of entries
	// so that we are guaranteed an empty slot (for termination).
	newN := n * 2                         // make it at most 50% full
	newN = 1 << sys.Len64(uint64(newN-1)) // round up to a power of 2

	// Allocate the new table.
	newSize := unsafe.Sizeof(abi.TypeAssertCache{}) + uintptr(newN-1)*unsafe.Sizeof(abi.TypeAssertCacheEntry{})
	newC := (*abi.TypeAssertCache)(mallocgc(newSize, nil, true))
	newC.Mask = uintptr(newN - 1)
	newEntries := unsafe.Slice(&newC.Entries[0], newN)

	// Fill the new table.
	addEntry := func(typ *_type, tab *itab) {
		h := int(typ.Hash) & (newN - 1)
		for {
			if newEntries[h].Typ == 0 {
				newEntries[h].Typ = uintptr(unsafe.Pointer(typ))
				newEntries[h].Itab = uintptr(unsafe.Pointer(tab))
				return
			}
			h = (h + 1) & (newN - 1)
		}
	}
	for _, e := range oldEntries {
		if e.Typ != 0 {
			addEntry((*_type)(unsafe.Pointer(e.Typ)), (*itab)(unsafe.Pointer(e.Itab)))
		}
	}
	addEntry(typ, tab)

	return newC
}

// Empty type assert cache. Contains one entry with a nil Typ (which
// causes a cache lookup to fail immediately.)
var emptyTypeAssertCache = abi.TypeAssertCache{Mask: 0}

// interfaceSwitch compares t against the list of cases in s.
// If t matches case i, interfaceSwitch returns the case index i and
// an itab for the pair <t, s.Cases[i]>.
// If there is no match, return N,nil, where N is the number
// of cases.
func interfaceSwitch(s *abi.InterfaceSwitch, t *_type) (int, *itab) {
	cases := unsafe.Slice(&s.Cases[0], s.NCases)

	// Results if we don't find a match.
	case_ := len(cases)
	var tab *itab

	// Look through each case in order.
	for i, c := range cases {
		tab = getitab(c, t, true)
		if tab != nil {
			case_ = i
			break
		}
	}

	if !abi.UseInterfaceSwitchCache(GOARCH) {
		return case_, tab
	}

	// Maybe update the cache, so the next time the generated code
	// doesn't need to call into the runtime.
	if cheaprand()&1023 != 0 {
		// Only bother updating the cache ~1 in 1000 times.
		// This ensures we don't waste memory on switches, or
		// switch arguments, that only happen a few times.
		return case_, tab
	}
	// Load the current cache.
	oldC := (*abi.InterfaceSwitchCache)(atomic.Loadp(unsafe.Pointer(&s.Cache)))

	if cheaprand()&uint32(oldC.Mask) != 0 {
		// As cache gets larger, choose to update it less often
		// so we can amortize the cost of building a new cache
		// (that cost is linear in oldc.Mask).
		return case_, tab
	}

	// Make a new cache.
	newC := buildInterfaceSwitchCache(oldC, t, case_, tab)

	// Update cache. Use compare-and-swap so if multiple threads
	// are fighting to update the cache, at least one of their
	// updates will stick.
	atomic_casPointer((*unsafe.Pointer)(unsafe.Pointer(&s.Cache)), unsafe.Pointer(oldC), unsafe.Pointer(newC))

	return case_, tab
}

// buildInterfaceSwitchCache constructs an interface switch cache
// containing all the entries from oldC plus the new entry
// (typ,case_,tab).
func buildInterfaceSwitchCache(oldC *abi.InterfaceSwitchCache, typ *_type, case_ int, tab *itab) *abi.InterfaceSwitchCache {
	oldEntries := unsafe.Slice(&oldC.Entries[0], oldC.Mask+1)

	// Count the number of entries we need.
	n := 1
	for _, e := range oldEntries {
		if e.Typ != 0 {
			n++
		}
	}

	// Figure out how big a table we need.
	// We need at least one more slot than the number of entries
	// so that we are guaranteed an empty slot (for termination).
	newN := n * 2                         // make it at most 50% full
	newN = 1 << sys.Len64(uint64(newN-1)) // round up to a power of 2

	// Allocate the new table.
	newSize := unsafe.Sizeof(abi.InterfaceSwitchCache{}) + uintptr(newN-1)*unsafe.Sizeof(abi.InterfaceSwitchCacheEntry{})
	newC := (*abi.InterfaceSwitchCache)(mallocgc(newSize, nil, true))
	newC.Mask = uintptr(newN - 1)
	newEntries := unsafe.Slice(&newC.Entries[0], newN)

	// Fill the new table.
	addEntry := func(typ *_type, case_ int, tab *itab) {
		h := int(typ.Hash) & (newN - 1)
		for {
			if newEntries[h].Typ == 0 {
				newEntries[h].Typ = uintptr(unsafe.Pointer(typ))
				newEntries[h].Case = case_
				newEntries[h].Itab = uintptr(unsafe.Pointer(tab))
				return
			}
			h = (h + 1) & (newN - 1)
		}
	}
	for _, e := range oldEntries {
		if e.Typ != 0 {
			addEntry((*_type)(unsafe.Pointer(e.Typ)), e.Case, (*itab)(unsafe.Pointer(e.Itab)))
		}
	}
	addEntry(typ, case_, tab)

	return newC
}

// Empty interface switch cache. Contains one entry with a nil Typ (which
// causes a cache lookup to fail immediately.)
var emptyInterfaceSwitchCache = abi.InterfaceSwitchCache{Mask: 0}

//go:linkname reflect_ifaceE2I reflect.ifaceE2I
func reflect_ifaceE2I(inter *interfacetype, e eface, dst *iface) {
	*dst = iface{assertE2I(inter, e._type), e.data}
}

//go:linkname reflectlite_ifaceE2I internal/reflectlite.ifaceE2I
func reflectlite_ifaceE2I(inter *interfacetype, e eface, dst *iface) {
	*dst = iface{assertE2I(inter, e._type), e.data}
}

func iterate_itabs(fn func(*itab)) {
	// Note: only runs during stop the world or with itabLock held,
	// so no other locks/atomics needed.
	t := itabTable
	for i := uintptr(0); i < t.size; i++ {
		m := *(**itab)(add(unsafe.Pointer(&t.entries), i*goarch.PtrSize))
		if m != nil {
			fn(m)
		}
	}
}

// staticuint64s is used to avoid allocating in convTx for small integer values.
var staticuint64s = [...]uint64{
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

// The linker redirects a reference of a method that it determined
// unreachable to a reference to this function, so it will throw if
// ever called.
func unreachableMethod() {
	throw("unreachable method called. linker bug?")
}

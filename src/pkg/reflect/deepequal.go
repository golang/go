// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Deep equality test via reflection

package reflect


// During deepValueEqual, must keep track of checks that are
// in progress.  The comparison algorithm assumes that all
// checks in progress are true when it reencounters them.
// Visited are stored in a map indexed by 17 * a1 + a2;
type visit struct {
	a1   uintptr
	a2   uintptr
	typ  Type
	next *visit
}

// Tests for deep equality using reflected types. The map argument tracks
// comparisons that have already been seen, which allows short circuiting on
// recursive types.
func deepValueEqual(v1, v2 Value, visited map[uintptr]*visit, depth int) bool {
	if v1 == nil || v2 == nil {
		return v1 == v2
	}
	if v1.Type() != v2.Type() {
		return false
	}

	// if depth > 10 { panic("deepValueEqual") }	// for debugging

	addr1 := v1.Addr()
	addr2 := v2.Addr()
	if addr1 > addr2 {
		// Canonicalize order to reduce number of entries in visited.
		addr1, addr2 = addr2, addr1
	}

	// Short circuit if references are identical ...
	if addr1 == addr2 {
		return true
	}

	// ... or already seen
	h := 17*addr1 + addr2
	seen := visited[h]
	typ := v1.Type()
	for p := seen; p != nil; p = p.next {
		if p.a1 == addr1 && p.a2 == addr2 && p.typ == typ {
			return true
		}
	}

	// Remember for later.
	visited[h] = &visit{addr1, addr2, typ, seen}

	switch v := v1.(type) {
	case *ArrayValue:
		arr1 := v
		arr2 := v2.(*ArrayValue)
		if arr1.Len() != arr2.Len() {
			return false
		}
		for i := 0; i < arr1.Len(); i++ {
			if !deepValueEqual(arr1.Elem(i), arr2.Elem(i), visited, depth+1) {
				return false
			}
		}
		return true
	case *SliceValue:
		arr1 := v
		arr2 := v2.(*SliceValue)
		if arr1.Len() != arr2.Len() {
			return false
		}
		for i := 0; i < arr1.Len(); i++ {
			if !deepValueEqual(arr1.Elem(i), arr2.Elem(i), visited, depth+1) {
				return false
			}
		}
		return true
	case *InterfaceValue:
		i1 := v.Interface()
		i2 := v2.Interface()
		if i1 == nil || i2 == nil {
			return i1 == i2
		}
		return deepValueEqual(NewValue(i1), NewValue(i2), visited, depth+1)
	case *PtrValue:
		return deepValueEqual(v.Elem(), v2.(*PtrValue).Elem(), visited, depth+1)
	case *StructValue:
		struct1 := v
		struct2 := v2.(*StructValue)
		for i, n := 0, v.NumField(); i < n; i++ {
			if !deepValueEqual(struct1.Field(i), struct2.Field(i), visited, depth+1) {
				return false
			}
		}
		return true
	case *MapValue:
		map1 := v
		map2 := v2.(*MapValue)
		if map1.Len() != map2.Len() {
			return false
		}
		for _, k := range map1.Keys() {
			if !deepValueEqual(map1.Elem(k), map2.Elem(k), visited, depth+1) {
				return false
			}
		}
		return true
	default:
		// Normal equality suffices
		return v1.Interface() == v2.Interface()
	}

	panic("Not reached")
}

// DeepEqual tests for deep equality. It uses normal == equality where possible
// but will scan members of arrays, slices, and fields of structs. It correctly
// handles recursive types.
func DeepEqual(a1, a2 interface{}) bool {
	if a1 == nil || a2 == nil {
		return a1 == a2
	}
	v1 := NewValue(a1)
	v2 := NewValue(a2)
	if v1.Type() != v2.Type() {
		return false
	}
	return deepValueEqual(v1, v2, make(map[uintptr]*visit), 0)
}

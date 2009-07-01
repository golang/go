// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Deep equality test via reflection

package reflect

import "reflect"

// During deepValueEqual, must keep track of checks that are
// in progress.  The comparison algorithm assumes that all
// checks in progress are true when it reencounters them.
// Visited are stored in a map indexed by 17 * a1 + a2;
type visit struct {
	a1 uintptr;
	a2 uintptr;
	typ Type;
	next *visit;
}

// Tests for deep equality using reflected types. The map argument tracks
// comparisons that have already been seen, which allows short circuiting on
// recursive types.
func deepValueEqual(v1, v2 Value, visited map[uintptr]*visit, depth int) bool {
	if v1 == nil {
		return v2 == nil
	}
	if v2 == nil {
		return false
	}
	if !equalType(v1.Type(), v2.Type()) {
		return false;
	}

	// if depth > 10 { panic("deepValueEqual") }	// for debugging

	addr1 := uintptr(v1.Addr());
	addr2 := uintptr(v2.Addr());
	if addr1 > addr2 {
		// Canonicalize order to reduce number of entries in visited.
		addr1, addr2 = addr2, addr1;
	}

	// Short circuit if references are identical ...
	if addr1 == addr2 {
		return true;
	}

	// ... or already seen
	h := 17 * addr1 + addr2;
	seen, ok := visited[h];
	typ := v1.Type();
	for p := seen; p != nil; p = p.next {
		if p.a1 == addr1 && p.a2 == addr2 && p.typ == typ {
			return true;
		}
	}

	// Remember for later.
	visited[h] = &visit{addr1, addr2, typ, seen};

	switch v1.Kind() {
	case ArrayKind:
		arr1 := v1.(ArrayValue);
		arr2 := v2.(ArrayValue);
		if arr1.IsSlice() != arr2.IsSlice() || arr1.Len() != arr2.Len() {
			return false;
		}
		for i := 0; i < arr1.Len(); i++ {
			if !deepValueEqual(arr1.Elem(i), arr2.Elem(i), visited, depth+1) {
				return false;
			}
		}
		return true;
	case InterfaceKind:
		i1 := v1.(InterfaceValue).Get();
		i2 := v2.(InterfaceValue).Get();
		if i1 == nil || i2 == nil {
			return i1 == i2;
		}
		return deepValueEqual(NewValue(i1), NewValue(i2), visited, depth+1);
	case MapKind:
		// TODO(dnadasi): Implement this fully once MapValue is implemented
		return v1.Interface() == v2.Interface();
	case PtrKind:
		return deepValueEqual(v1.(PtrValue).Sub(), v2.(PtrValue).Sub(), visited, depth+1);
	case StructKind:
		struct1 := v1.(StructValue);
		struct2 := v2.(StructValue);
		if struct1.Len() != struct2.Len() {
			return false;
		}
		for i := 0; i < struct1.Len(); i++ {
			if !deepValueEqual(struct1.Field(i), struct2.Field(i), visited, depth+1) {
				return false;
			}
		}
		return true;
	default:
		// Normal equality suffices
		return v1.Interface() == v2.Interface();
	}

	panic("Not reached");
}

// DeepEqual tests for deep equality. It uses normal == equality where possible
// but will scan members of arrays, slices, and fields of structs. It correctly
// handles recursive types. Until reflection supports maps, maps are equal iff
// they are identical.
func DeepEqual(a1, a2 interface{}) bool {
	v1 := NewValue(a1);
	v2 := NewValue(a2);
	if !equalType(v1.Type(), v2.Type()) {
		return false;
	}
	return deepValueEqual(v1, v2, make(map[uintptr]*visit), 0);
}

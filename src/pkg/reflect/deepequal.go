// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Deep equality test via reflection

package reflect

import "reflect"

// Tests for deep equality using reflected types. The map argument tracks
// comparisons that have already been seen, which allows short circuiting on
// recursive types.
func deepValueEqual(v1, v2 Value, visited map[Addr]Addr, depth int) bool {
	if v1 == nil {
		return v2 == nil
	}
	if v2 == nil {
		return false
	}
	if v1.Kind() != v2.Kind() {
		return false;
	}

	// if depth > 10 { panic("deepValueEqual") }	// for debugging

	// Short circuit if references are identical or already seen
	addr1 := v1.Addr();
	addr2 := v2.Addr();

	if addr1 == addr2 {
		return true;
	}
	if vaddr, ok := visited[addr1]; ok && vaddr == addr2 {
		return true;
	}
	visited[addr1] = addr2;

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
	return deepValueEqual(v1, v2, make(map[Addr]Addr), 0);
}

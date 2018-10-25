// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package fmtsort provides a general stable ordering mechanism
// for maps, on behalf of the fmt and text/template packages.
// It is not guaranteed to be efficient and works only for types
// that are valid map keys.
package fmtsort

import (
	"reflect"
	"sort"
)

// Note: Throughout this package we avoid calling reflect.Value.Interface as
// it is not always legal to do so and it's easier to avoid the issue than to face it.

// SortedMap represents a map's keys and values. The keys and values are
// aligned in index order: Value[i] is the value in the map corresponding to Key[i].
type SortedMap struct {
	Key   []reflect.Value
	Value []reflect.Value
}

func (o *SortedMap) Len() int           { return len(o.Key) }
func (o *SortedMap) Less(i, j int) bool { return compare(o.Key[i], o.Key[j]) < 0 }
func (o *SortedMap) Swap(i, j int) {
	o.Key[i], o.Key[j] = o.Key[j], o.Key[i]
	o.Value[i], o.Value[j] = o.Value[j], o.Value[i]
}

// Sort accepts a map and returns a SortedMap that has the same keys and
// values but in a stable sorted order according to the keys, modulo issues
// raised by unorderable key values such as NaNs.
//
// The ordering rules are more general than with Go's < operator:
//
//  - when applicable, nil compares low
//  - ints, floats, and strings order by <
//  - NaN compares less than non-NaN floats
//  - bool compares false before true
//  - complex compares real, then imag
//  - pointers compare by machine address
//  - channel values compare by machine address
//  - structs compare each field in turn
//  - arrays compare each element in turn.
//    Otherwise identical arrays compare by length.
//  - interface values compare first by reflect.Type describing the concrete type
//    and then by concrete value as described in the previous rules.
//
func Sort(mapValue reflect.Value) *SortedMap {
	if mapValue.Type().Kind() != reflect.Map {
		return nil
	}
	key := make([]reflect.Value, mapValue.Len())
	value := make([]reflect.Value, len(key))
	iter := mapValue.MapRange()
	for i := 0; iter.Next(); i++ {
		key[i] = iter.Key()
		value[i] = iter.Value()
	}
	sorted := &SortedMap{
		Key:   key,
		Value: value,
	}
	sort.Stable(sorted)
	return sorted
}

// compare compares two values of the same type. It returns -1, 0, 1
// according to whether a > b (1), a == b (0), or a < b (-1).
// If the types differ, it returns -1.
// See the comment on Sort for the comparison rules.
func compare(aVal, bVal reflect.Value) int {
	aType, bType := aVal.Type(), bVal.Type()
	if aType != bType {
		return -1 // No good answer possible, but don't return 0: they're not equal.
	}
	switch aVal.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		a, b := aVal.Int(), bVal.Int()
		switch {
		case a < b:
			return -1
		case a > b:
			return 1
		default:
			return 0
		}
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		a, b := aVal.Uint(), bVal.Uint()
		switch {
		case a < b:
			return -1
		case a > b:
			return 1
		default:
			return 0
		}
	case reflect.String:
		a, b := aVal.String(), bVal.String()
		switch {
		case a < b:
			return -1
		case a > b:
			return 1
		default:
			return 0
		}
	case reflect.Float32, reflect.Float64:
		return floatCompare(aVal.Float(), bVal.Float())
	case reflect.Complex64, reflect.Complex128:
		a, b := aVal.Complex(), bVal.Complex()
		if c := floatCompare(real(a), real(b)); c != 0 {
			return c
		}
		return floatCompare(imag(a), imag(b))
	case reflect.Bool:
		a, b := aVal.Bool(), bVal.Bool()
		switch {
		case a == b:
			return 0
		case a:
			return 1
		default:
			return -1
		}
	case reflect.Ptr:
		a, b := aVal.Pointer(), bVal.Pointer()
		switch {
		case a < b:
			return -1
		case a > b:
			return 1
		default:
			return 0
		}
	case reflect.Chan:
		if c, ok := nilCompare(aVal, bVal); ok {
			return c
		}
		ap, bp := aVal.Pointer(), bVal.Pointer()
		switch {
		case ap < bp:
			return -1
		case ap > bp:
			return 1
		default:
			return 0
		}
	case reflect.Struct:
		for i := 0; i < aVal.NumField(); i++ {
			if c := compare(aVal.Field(i), bVal.Field(i)); c != 0 {
				return c
			}
		}
		return 0
	case reflect.Array:
		for i := 0; i < aVal.Len(); i++ {
			if c := compare(aVal.Index(i), bVal.Index(i)); c != 0 {
				return c
			}
		}
		return 0
	case reflect.Interface:
		if c, ok := nilCompare(aVal, bVal); ok {
			return c
		}
		c := compare(reflect.ValueOf(aType), reflect.ValueOf(bType))
		if c != 0 {
			return c
		}
		return compare(aVal.Elem(), bVal.Elem())
	default:
		// Certain types cannot appear as keys (maps, funcs, slices), but be explicit.
		panic("bad type in compare: " + aType.String())
	}
}

// nilCompare checks whether either value is nil. If not, the boolean is false.
// If either value is nil, the boolean is true and the integer is the comparison
// value. The comparison is defined to be 0 if both are nil, otherwise the one
// nil value compares low. Both arguments must represent a chan, func,
// interface, map, pointer, or slice.
func nilCompare(aVal, bVal reflect.Value) (int, bool) {
	if aVal.IsNil() {
		if bVal.IsNil() {
			return 0, true
		}
		return -1, true
	}
	if bVal.IsNil() {
		return 1, true
	}
	return 0, false
}

// floatCompare compares two floating-point values. NaNs compare low.
func floatCompare(a, b float64) int {
	switch {
	case isNaN(a):
		return -1 // No good answer if b is a NaN so don't bother checking.
	case isNaN(b):
		return 1
	case a < b:
		return -1
	case a > b:
		return 1
	}
	return 0
}

func isNaN(a float64) bool {
	return a != a
}

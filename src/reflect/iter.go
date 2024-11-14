// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package reflect

import "iter"

func rangeNum[T int8 | int16 | int32 | int64 | int |
	uint8 | uint16 | uint32 | uint64 | uint |
	uintptr, N int64 | uint64](v N) iter.Seq[Value] {
	return func(yield func(v Value) bool) {
		// cannot use range T(v) because no core type.
		for i := T(0); i < T(v); i++ {
			if !yield(ValueOf(i)) {
				return
			}
		}
	}
}

// Seq returns an iter.Seq[Value] that loops over the elements of v.
// If v's kind is Func, it must be a function that has no results and
// that takes a single argument of type func(T) bool for some type T.
// If v's kind is Pointer, the pointer element type must have kind Array.
// Otherwise v's kind must be Int, Int8, Int16, Int32, Int64,
// Uint, Uint8, Uint16, Uint32, Uint64, Uintptr,
// Array, Chan, Map, Slice, or String.
func (v Value) Seq() iter.Seq[Value] {
	if canRangeFunc(v.typ()) {
		return func(yield func(Value) bool) {
			rf := MakeFunc(v.Type().In(0), func { in -> []Value{ValueOf(yield(in[0]))} })
			v.Call([]Value{rf})
		}
	}
	switch v.Kind() {
	case Int:
		return rangeNum[int](v.Int())
	case Int8:
		return rangeNum[int8](v.Int())
	case Int16:
		return rangeNum[int16](v.Int())
	case Int32:
		return rangeNum[int32](v.Int())
	case Int64:
		return rangeNum[int64](v.Int())
	case Uint:
		return rangeNum[uint](v.Uint())
	case Uint8:
		return rangeNum[uint8](v.Uint())
	case Uint16:
		return rangeNum[uint16](v.Uint())
	case Uint32:
		return rangeNum[uint32](v.Uint())
	case Uint64:
		return rangeNum[uint64](v.Uint())
	case Uintptr:
		return rangeNum[uintptr](v.Uint())
	case Pointer:
		if v.Elem().kind() != Array {
			break
		}
		return func(yield func(Value) bool) {
			v = v.Elem()
			for i := range v.Len() {
				if !yield(ValueOf(i)) {
					return
				}
			}
		}
	case Array, Slice:
		return func(yield func(Value) bool) {
			for i := range v.Len() {
				if !yield(ValueOf(i)) {
					return
				}
			}
		}
	case String:
		return func(yield func(Value) bool) {
			for i := range v.String() {
				if !yield(ValueOf(i)) {
					return
				}
			}
		}
	case Map:
		return func(yield func(Value) bool) {
			i := v.MapRange()
			for i.Next() {
				if !yield(i.Key()) {
					return
				}
			}
		}
	case Chan:
		return func(yield func(Value) bool) {
			for value, ok := v.Recv(); ok; value, ok = v.Recv() {
				if !yield(value) {
					return
				}
			}
		}
	}
	panic("reflect: " + v.Type().String() + " cannot produce iter.Seq[Value]")
}

// Seq2 returns an iter.Seq2[Value, Value] that loops over the elements of v.
// If v's kind is Func, it must be a function that has no results and
// that takes a single argument of type func(K, V) bool for some type K, V.
// If v's kind is Pointer, the pointer element type must have kind Array.
// Otherwise v's kind must be Array, Map, Slice, or String.
func (v Value) Seq2() iter.Seq2[Value, Value] {
	if canRangeFunc2(v.typ()) {
		return func(yield func(Value, Value) bool) {
			rf := MakeFunc(v.Type().In(0), func { in -> []Value{ValueOf(yield(in[0], in[1]))} })
			v.Call([]Value{rf})
		}
	}
	switch v.Kind() {
	case Pointer:
		if v.Elem().kind() != Array {
			break
		}
		return func(yield func(Value, Value) bool) {
			v = v.Elem()
			for i := range v.Len() {
				if !yield(ValueOf(i), v.Index(i)) {
					return
				}
			}
		}
	case Array, Slice:
		return func(yield func(Value, Value) bool) {
			for i := range v.Len() {
				if !yield(ValueOf(i), v.Index(i)) {
					return
				}
			}
		}
	case String:
		return func(yield func(Value, Value) bool) {
			for i, v := range v.String() {
				if !yield(ValueOf(i), ValueOf(v)) {
					return
				}
			}
		}
	case Map:
		return func(yield func(Value, Value) bool) {
			i := v.MapRange()
			for i.Next() {
				if !yield(i.Key(), i.Value()) {
					return
				}
			}
		}
	}
	panic("reflect: " + v.Type().String() + " cannot produce iter.Seq2[Value, Value]")
}

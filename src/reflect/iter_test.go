// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package reflect_test

import (
	"iter"
	"maps"
	. "reflect"
	"testing"
)

func TestValueSeq(t *testing.T) {
	m := map[string]int{
		"1": 1,
		"2": 2,
		"3": 3,
		"4": 4,
	}
	c := make(chan int, 3)
	for i := range 3 {
		c <- i
	}
	close(c)
	tests := []struct {
		name  string
		val   Value
		check func(*testing.T, iter.Seq[Value])
	}{
		{"int", ValueOf(4), func(t *testing.T, s iter.Seq[Value]) {
			i := int64(0)
			for v := range s {
				if v.Int() != i {
					t.Fatalf("got %d, want %d", v.Int(), i)
				}
				i++
			}
			if i != 4 {
				t.Fatalf("should loop four times")
			}
		}},
		{"int8", ValueOf(int8(4)), func(t *testing.T, s iter.Seq[Value]) {
			i := int8(0)
			for v := range s {
				if v.Interface().(int8) != i {
					t.Fatalf("got %d, want %d", v.Int(), i)
				}
				i++
			}
			if i != 4 {
				t.Fatalf("should loop four times")
			}
		}},
		{"uint", ValueOf(uint64(4)), func(t *testing.T, s iter.Seq[Value]) {
			i := uint64(0)
			for v := range s {
				if v.Uint() != i {
					t.Fatalf("got %d, want %d", v.Uint(), i)
				}
				i++
			}
			if i != 4 {
				t.Fatalf("should loop four times")
			}
		}},
		{"uint8", ValueOf(uint8(4)), func(t *testing.T, s iter.Seq[Value]) {
			i := uint8(0)
			for v := range s {
				if v.Interface().(uint8) != i {
					t.Fatalf("got %d, want %d", v.Int(), i)
				}
				i++
			}
			if i != 4 {
				t.Fatalf("should loop four times")
			}
		}},
		{"*[4]int", ValueOf(&[4]int{1, 2, 3, 4}), func(t *testing.T, s iter.Seq[Value]) {
			i := int64(0)
			for v := range s {
				if v.Int() != i {
					t.Fatalf("got %d, want %d", v.Int(), i)
				}
				i++
			}
			if i != 4 {
				t.Fatalf("should loop four times")
			}
		}},
		{"[4]int", ValueOf([4]int{1, 2, 3, 4}), func(t *testing.T, s iter.Seq[Value]) {
			i := int64(0)
			for v := range s {
				if v.Int() != i {
					t.Fatalf("got %d, want %d", v.Int(), i)
				}
				i++
			}
			if i != 4 {
				t.Fatalf("should loop four times")
			}
		}},
		{"[]int", ValueOf([]int{1, 2, 3, 4}), func(t *testing.T, s iter.Seq[Value]) {
			i := int64(0)
			for v := range s {
				if v.Int() != i {
					t.Fatalf("got %d, want %d", v.Int(), i)
				}
				i++
			}
			if i != 4 {
				t.Fatalf("should loop four times")
			}
		}},
		{"string", ValueOf("12语言"), func(t *testing.T, s iter.Seq[Value]) {
			i := int64(0)
			indexes := []int64{0, 1, 2, 5}
			for v := range s {
				if v.Int() != indexes[i] {
					t.Fatalf("got %d, want %d", v.Int(), indexes[i])
				}
				i++
			}
			if i != 4 {
				t.Fatalf("should loop four times")
			}
		}},
		{"map[string]int", ValueOf(m), func(t *testing.T, s iter.Seq[Value]) {
			copy := maps.Clone(m)
			for v := range s {
				if _, ok := copy[v.String()]; !ok {
					t.Fatalf("unexpected %v", v.Interface())
				}
				delete(copy, v.String())
			}
			if len(copy) != 0 {
				t.Fatalf("should loop four times")
			}
		}},
		{"chan int", ValueOf(c), func(t *testing.T, s iter.Seq[Value]) {
			i := 0
			m := map[int64]bool{
				0: false,
				1: false,
				2: false,
			}
			for v := range s {
				if b, ok := m[v.Int()]; !ok || b {
					t.Fatalf("unexpected %v", v.Interface())
				}
				m[v.Int()] = true
				i++
			}
			if i != 3 {
				t.Fatalf("should loop three times")
			}
		}},
		{"func", ValueOf(func(yield func(int) bool) {
			for i := range 4 {
				if !yield(i) {
					return
				}
			}
		}), func(t *testing.T, s iter.Seq[Value]) {
			i := int64(0)
			for v := range s {
				if v.Int() != i {
					t.Fatalf("got %d, want %d", v.Int(), i)
				}
				i++
			}
			if i != 4 {
				t.Fatalf("should loop four times")
			}
		}},
		{"method", ValueOf(methodIter{}).Method(0), func(t *testing.T, s iter.Seq[Value]) {
			i := int64(0)
			for v := range s {
				if v.Int() != i {
					t.Fatalf("got %d, want %d", v.Int(), i)
				}
				i++
			}
			if i != 4 {
				t.Fatalf("should loop four times")
			}
		}},
	}
	for _, tc := range tests {
		seq := tc.val.Seq()
		tc.check(t, seq)
	}
}

func TestValueSeq2(t *testing.T) {
	m := map[string]int{
		"1": 1,
		"2": 2,
		"3": 3,
		"4": 4,
	}
	tests := []struct {
		name  string
		val   Value
		check func(*testing.T, iter.Seq2[Value, Value])
	}{
		{"*[4]int", ValueOf(&[4]int{1, 2, 3, 4}), func(t *testing.T, s iter.Seq2[Value, Value]) {
			i := int64(0)
			for v1, v2 := range s {
				if v1.Int() != i {
					t.Fatalf("got %d, want %d", v1.Int(), i)
				}
				i++
				if v2.Int() != i {
					t.Fatalf("got %d, want %d", v2.Int(), i)
				}
			}
			if i != 4 {
				t.Fatalf("should loop four times")
			}
		}},
		{"[4]int", ValueOf([4]int{1, 2, 3, 4}), func(t *testing.T, s iter.Seq2[Value, Value]) {
			i := int64(0)
			for v1, v2 := range s {
				if v1.Int() != i {
					t.Fatalf("got %d, want %d", v1.Int(), i)
				}
				i++
				if v2.Int() != i {
					t.Fatalf("got %d, want %d", v2.Int(), i)
				}
			}
			if i != 4 {
				t.Fatalf("should loop four times")
			}
		}},
		{"[]int", ValueOf([]int{1, 2, 3, 4}), func(t *testing.T, s iter.Seq2[Value, Value]) {
			i := int64(0)
			for v1, v2 := range s {
				if v1.Int() != i {
					t.Fatalf("got %d, want %d", v1.Int(), i)
				}
				i++
				if v2.Int() != i {
					t.Fatalf("got %d, want %d", v2.Int(), i)
				}
			}
			if i != 4 {
				t.Fatalf("should loop four times")
			}
		}},
		{"string", ValueOf("12语言"), func(t *testing.T, s iter.Seq2[Value, Value]) {
			next, stop := iter.Pull2(s)
			defer stop()
			i := int64(0)
			for j, s := range "12语言" {
				v1, v2, ok := next()
				if !ok {
					t.Fatalf("should loop four times")
				}
				if v1.Int() != int64(j) {
					t.Fatalf("got %d, want %d", v1.Int(), j)
				}
				if v2.Interface() != s {
					t.Fatalf("got %v, want %v", v2.Interface(), s)
				}
				i++
			}
			if i != 4 {
				t.Fatalf("should loop four times")
			}
		}},
		{"map[string]int", ValueOf(m), func(t *testing.T, s iter.Seq2[Value, Value]) {
			copy := maps.Clone(m)
			for v1, v2 := range s {
				v, ok := copy[v1.String()]
				if !ok {
					t.Fatalf("unexpected %v", v1.String())
				}
				if v != v2.Interface() {
					t.Fatalf("got %v, want %d", v2.Interface(), v)
				}
				delete(copy, v1.String())
			}
			if len(copy) != 0 {
				t.Fatalf("should loop four times")
			}
		}},
		{"func", ValueOf(func(f func(int, int) bool) {
			for i := range 4 {
				f(i, i+1)
			}
		}), func(t *testing.T, s iter.Seq2[Value, Value]) {
			i := int64(0)
			for v1, v2 := range s {
				if v1.Int() != i {
					t.Fatalf("got %d, want %d", v1.Int(), i)
				}
				i++
				if v2.Int() != i {
					t.Fatalf("got %d, want %d", v2.Int(), i)
				}
			}
			if i != 4 {
				t.Fatalf("should loop four times")
			}
		}},
		{"method", ValueOf(methodIter2{}).Method(0), func(t *testing.T, s iter.Seq2[Value, Value]) {
			i := int64(0)
			for v1, v2 := range s {
				if v1.Int() != i {
					t.Fatalf("got %d, want %d", v1.Int(), i)
				}
				i++
				if v2.Int() != i {
					t.Fatalf("got %d, want %d", v2.Int(), i)
				}
			}
			if i != 4 {
				t.Fatalf("should loop four times")
			}
		}},
	}
	for _, tc := range tests {
		seq := tc.val.Seq2()
		tc.check(t, seq)
	}
}

// methodIter is a type from which we can derive a method
// value that is an iter.Seq.
type methodIter struct{}

func (methodIter) Seq(yield func(int) bool) {
	for i := range 4 {
		if !yield(i) {
			return
		}
	}
}

// methodIter2 is a type from which we can derive a method
// value that is an iter.Seq2.
type methodIter2 struct{}

func (methodIter2) Seq2(yield func(int, int) bool) {
	for i := range 4 {
		if !yield(i, i+1) {
			return
		}
	}
}

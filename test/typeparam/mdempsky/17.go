// run

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that implicit conversions of derived types to interface type
// in range loops work correctly.

package main

import (
	"fmt"
	"reflect"
)

func main() {
	test{"int", "V"}.match(RangeArrayAny[V]())
	test{"int", "V"}.match(RangeArrayIface[V]())
	test{"V"}.match(RangeChanAny[V]())
	test{"V"}.match(RangeChanIface[V]())
	test{"K", "V"}.match(RangeMapAny[K, V]())
	test{"K", "V"}.match(RangeMapIface[K, V]())
	test{"int", "V"}.match(RangeSliceAny[V]())
	test{"int", "V"}.match(RangeSliceIface[V]())
}

type test []string

func (t test) match(args ...any) {
	if len(t) != len(args) {
		fmt.Printf("FAIL: want %v values, have %v\n", len(t), len(args))
		return
	}
	for i, want := range t {
		if have := reflect.TypeOf(args[i]).Name(); want != have {
			fmt.Printf("FAIL: %v: want type %v, have %v\n", i, want, have)
		}
	}
}

type iface interface{ M() int }

type K int
type V int

func (K) M() int { return 0 }
func (V) M() int { return 0 }

func RangeArrayAny[V any]() (k, v any) {
	for k, v = range [...]V{zero[V]()} {
	}
	return
}

func RangeArrayIface[V iface]() (k any, v iface) {
	for k, v = range [...]V{zero[V]()} {
	}
	return
}

func RangeChanAny[V any]() (v any) {
	for v = range chanOf(zero[V]()) {
	}
	return
}

func RangeChanIface[V iface]() (v iface) {
	for v = range chanOf(zero[V]()) {
	}
	return
}

func RangeMapAny[K comparable, V any]() (k, v any) {
	for k, v = range map[K]V{zero[K](): zero[V]()} {
	}
	return
}

func RangeMapIface[K interface {
	iface
	comparable
}, V iface]() (k, v iface) {
	for k, v = range map[K]V{zero[K](): zero[V]()} {
	}
	return
}

func RangeSliceAny[V any]() (k, v any) {
	for k, v = range []V{zero[V]()} {
	}
	return
}

func RangeSliceIface[V iface]() (k any, v iface) {
	for k, v = range []V{zero[V]()} {
	}
	return
}

func chanOf[T any](elems ...T) chan T {
	c := make(chan T, len(elems))
	for _, elem := range elems {
		c <- elem
	}
	close(c)
	return c
}

func zero[T any]() (_ T) { return }

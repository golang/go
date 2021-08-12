// run -gcflags=-G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "fmt"

type value[T any] struct {
	val T
}

func get[T any](v *value[T]) T {
	return v.val
}

func set[T any](v *value[T], val T) {
	v.val = val
}

func (v *value[T]) set(val T) {
	v.val = val
}

func (v *value[T]) get() T {
	return v.val
}

func main() {
	var v1 value[int]
	set(&v1, 1)
	if got, want := get(&v1), 1; got != want {
		panic(fmt.Sprintf("get() == %d, want %d", got, want))
	}

	v1.set(2)
	if got, want := v1.get(), 2; got != want {
		panic(fmt.Sprintf("get() == %d, want %d", got, want))
	}

	v1p := new(value[int])
	set(v1p, 3)
	if got, want := get(v1p), 3; got != want {
		panic(fmt.Sprintf("get() == %d, want %d", got, want))
	}

	v1p.set(4)
	if got, want := v1p.get(), 4; got != want {
		panic(fmt.Sprintf("get() == %d, want %d", got, want))
	}

	var v2 value[string]
	set(&v2, "a")
	if got, want := get(&v2), "a"; got != want {
		panic(fmt.Sprintf("get() == %q, want %q", got, want))
	}

	v2.set("b")
	if got, want := get(&v2), "b"; got != want {
		panic(fmt.Sprintf("get() == %q, want %q", got, want))
	}

	v2p := new(value[string])
	set(v2p, "c")
	if got, want := get(v2p), "c"; got != want {
		panic(fmt.Sprintf("get() == %d, want %d", got, want))
	}

	v2p.set("d")
	if got, want := v2p.get(), "d"; got != want {
		panic(fmt.Sprintf("get() == %d, want %d", got, want))
	}
}

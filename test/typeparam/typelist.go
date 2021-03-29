// compile -G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file tests type lists & structural constraints.

package p

// Assignability of an unnamed pointer type to a type parameter that
// has a matching underlying type.
func _[T interface{}, PT interface{type *T}] (x T) PT {
    return &x
}

// Indexing of generic types containing type parameters in their type list:
func at[T interface{ type []E }, E any](x T, i int) E {
        return x[i]
}

// A generic type inside a function acts like a named type. Its underlying
// type is itself, its "operational type" is defined by the type list in
// the tybe bound, if any.
func _[T interface{type int}](x T) {
	type myint int
	var _ int = int(x)
	var _ T = 42
	var _ T = T(myint(42))
}

// Indexing a generic type which has a structural contraints to be an array.
func _[T interface { type [10]int }](x T) {
	_ = x[9] // ok
}

// Dereference of a generic type which has a structural contraint to be a pointer.
func _[T interface{ type *int }](p T) int {
	return *p
}

// Channel send and receive on a generic type which has a structural constraint to
// be a channel.
func _[T interface{ type chan int }](ch T) int {
	// This would deadlock if executed (but ok for a compile test)
	ch <- 0
	return <- ch
}

// Calling of a generic type which has a structural constraint to be a function.
func _[T interface{ type func() }](f T) {
	f()
	go f()
}

// Same, but function has a parameter and return value.
func _[T interface{ type func(string) int }](f T) int {
	return f("hello")
}

// Map access of a generic type which has a structural constraint to be a map.
func _[V any, T interface { type map[string]V }](p T) V {
	return p["test"]
}

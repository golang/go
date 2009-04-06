// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The iterable package provides several traversal and searching methods.
// It can be used on anything that satisfies the Iterable interface,
// including vector, though certain functions, such as Map, can also be used on
// something that would produce an infinite amount of data.
package iterable

import "vector"


type Iterable interface {
	// Iter should return a fresh channel each time it is called.
	Iter() <-chan interface {}
}


// All tests whether f is true for every element of iter.
func All(iter Iterable, f func(e interface {}) bool) bool {
	for e := range iter.Iter() {
		if !f(e) {
			return false
		}
	}
	return true
}


// Any tests whether f is true for at least one element of iter.
func Any(iter Iterable, f func(e interface {}) bool) bool {
	return !All(iter, func(e interface {}) bool { return !f(e) });
}


// Data returns a slice containing the elements of iter.
func Data(iter Iterable) []interface {} {
	vec := vector.New(0);
	for e := range iter.Iter() {
		vec.Push(e)
	}
	return vec.Data()
}


// mappedIterable is a helper struct that implements Iterable, returned by Map.
type mappedIterable struct {
	it Iterable;
	f func(interface {}) interface {};
}


func (m mappedIterable) iterate(out chan<- interface {}) {
	for e := range m.it.Iter() {
		out <- m.f(e)
	}
	close(out)
}


func (m mappedIterable) Iter() <-chan interface {} {
	ch := make(chan interface {});
	go m.iterate(ch);
	return ch;
}


// Map returns an Iterable that returns the result of applying f to each
// element of iter.
func Map(iter Iterable, f func(e interface {}) interface {}) Iterable {
	return mappedIterable{ iter, f }
}


// TODO:
// - Find, Filter
// - Inject
// - Partition
// - Zip

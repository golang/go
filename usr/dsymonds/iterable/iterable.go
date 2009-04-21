// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The iterable package provides several traversal and searching methods.
// It can be used on anything that satisfies the Iterable interface,
// including vector, though certain functions, such as Map, can also be used on
// something that would produce an infinite amount of data.
package iterable

import "container/vector"

type Iterable interface {
	// Iter should return a fresh channel each time it is called.
	Iter() <-chan interface {}
}

func not(f func(interface {}) bool) (func(interface {}) bool) {
	return func(e interface {}) bool { return !f(e) }
}

// All tests whether f is true for every element of iter.
func All(iter Iterable, f func(interface {}) bool) bool {
	for e := range iter.Iter() {
		if !f(e) {
			return false
		}
	}
	return true
}

// Any tests whether f is true for at least one element of iter.
func Any(iter Iterable, f func(interface {}) bool) bool {
	return !All(iter, not(f))
}

// Data returns a slice containing the elements of iter.
func Data(iter Iterable) []interface {} {
	vec := vector.New(0);
	for e := range iter.Iter() {
		vec.Push(e)
	}
	return vec.Data()
}

// filteredIterable is a struct that implements Iterable with each element
// passed through a filter.
type filteredIterable struct {
	it Iterable;
	f func(interface {}) bool;
}

func (f *filteredIterable) iterate(out chan<- interface {}) {
	for e := range f.it.Iter() {
		if f.f(e) {
			out <- e
		}
	}
	close(out)
}

func (f *filteredIterable) Iter() <-chan interface {} {
	ch := make(chan interface {});
	go f.iterate(ch);
	return ch;
}

// Filter returns an Iterable that returns the elements of iter that satisfy f.
func Filter(iter Iterable, f func(interface {}) bool) Iterable {
	return &filteredIterable{ iter, f }
}

// Find returns the first element of iter that satisfies f.
// Returns nil if no such element is found.
func Find(iter Iterable, f func(interface {}) bool) interface {} {
	for e := range Filter(iter, f).Iter() {
		return e
	}
	return nil
}

// Injector is a type representing a function that takes two arguments,
// an accumulated value and an element, and returns the next accumulated value.
// See the Inject function.
type Injector func(interface {}, interface {}) interface{};

// Inject combines the elements of iter by repeatedly calling f with an
// accumulated value and each element in order. The starting accumulated value
// is initial, and after each call the accumulated value is set to the return
// value of f. For instance, to compute a sum:
//   var arr IntArray = []int{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
//   sum := iterable.Inject(arr, 0,
//                          func(ax interface {}, x interface {}) interface {} {
//                            return ax.(int) + x.(int) }).(int)
func Inject(iter Iterable, initial interface {}, f Injector) interface {} {
	acc := initial;
	for e := range iter.Iter() {
		acc = f(acc, e)
	}
	return acc
}

// mappedIterable is a helper struct that implements Iterable, returned by Map.
type mappedIterable struct {
	it Iterable;
	f func(interface {}) interface {};
}

func (m *mappedIterable) iterate(out chan<- interface {}) {
	for e := range m.it.Iter() {
		out <- m.f(e)
	}
	close(out)
}

func (m *mappedIterable) Iter() <-chan interface {} {
	ch := make(chan interface {});
	go m.iterate(ch);
	return ch
}

// Map returns an Iterable that returns the result of applying f to each
// element of iter.
func Map(iter Iterable, f func(interface {}) interface {}) Iterable {
	return &mappedIterable{ iter, f }
}

// Partition(iter, f) returns Filter(iter, f) and Filter(iter, !f).
func Partition(iter Iterable, f func(interface {}) bool) (Iterable, Iterable)  {
	return Filter(iter, f), Filter(iter, not(f))
}

// TODO:
// - Zip

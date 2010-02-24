// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The iterable package provides several traversal and searching methods.
// It can be used on anything that satisfies the Iterable interface,
// including vector, though certain functions, such as Map, can also be used on
// something that would produce an infinite amount of data.
package iterable

import (
	"container/list"
	"container/vector"
)

type Iterable interface {
	// Iter should return a fresh channel each time it is called.
	Iter() <-chan interface{}
}

func not(f func(interface{}) bool) func(interface{}) bool {
	return func(e interface{}) bool { return !f(e) }
}

// All tests whether f is true for every element of iter.
func All(iter Iterable, f func(interface{}) bool) bool {
	for e := range iter.Iter() {
		if !f(e) {
			return false
		}
	}
	return true
}

// Any tests whether f is true for at least one element of iter.
func Any(iter Iterable, f func(interface{}) bool) bool {
	return !All(iter, not(f))
}

// Data returns a slice containing the elements of iter.
func Data(iter Iterable) []interface{} {
	vec := new(vector.Vector)
	for e := range iter.Iter() {
		vec.Push(e)
	}
	return vec.Data()
}

// filteredIterable is a struct that implements Iterable with each element
// passed through a filter.
type filteredIterable struct {
	it Iterable
	f  func(interface{}) bool
}

func (f *filteredIterable) iterate(out chan<- interface{}) {
	for e := range f.it.Iter() {
		if f.f(e) {
			out <- e
		}
	}
	close(out)
}

func (f *filteredIterable) Iter() <-chan interface{} {
	ch := make(chan interface{})
	go f.iterate(ch)
	return ch
}

// Filter returns an Iterable that returns the elements of iter that satisfy f.
func Filter(iter Iterable, f func(interface{}) bool) Iterable {
	return &filteredIterable{iter, f}
}

// Find returns the first element of iter that satisfies f.
// Returns nil if no such element is found.
func Find(iter Iterable, f func(interface{}) bool) interface{} {
	for e := range Filter(iter, f).Iter() {
		return e
	}
	return nil
}

// Injector is a type representing a function that takes two arguments,
// an accumulated value and an element, and returns the next accumulated value.
// See the Inject function.
type Injector func(interface{}, interface{}) interface{}

// Inject combines the elements of iter by repeatedly calling f with an
// accumulated value and each element in order. The starting accumulated value
// is initial, and after each call the accumulated value is set to the return
// value of f. For instance, to compute a sum:
//   var arr IntArray = []int{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
//   sum := iterable.Inject(arr, 0,
//                          func(ax interface {}, x interface {}) interface {} {
//                            return ax.(int) + x.(int) }).(int)
func Inject(iter Iterable, initial interface{}, f Injector) interface{} {
	acc := initial
	for e := range iter.Iter() {
		acc = f(acc, e)
	}
	return acc
}

// mappedIterable is a helper struct that implements Iterable, returned by Map.
type mappedIterable struct {
	it Iterable
	f  func(interface{}) interface{}
}

func (m *mappedIterable) iterate(out chan<- interface{}) {
	for e := range m.it.Iter() {
		out <- m.f(e)
	}
	close(out)
}

func (m *mappedIterable) Iter() <-chan interface{} {
	ch := make(chan interface{})
	go m.iterate(ch)
	return ch
}

// Map returns an Iterable that returns the result of applying f to each
// element of iter.
func Map(iter Iterable, f func(interface{}) interface{}) Iterable {
	return &mappedIterable{iter, f}
}

// Partition(iter, f) returns Filter(iter, f) and Filter(iter, !f).
func Partition(iter Iterable, f func(interface{}) bool) (Iterable, Iterable) {
	return Filter(iter, f), Filter(iter, not(f))
}

// helper type for the Take/TakeWhile/Drop/DropWhile functions.
// primarily used so that the .Iter() method can be attached
type iterFunc func(chan<- interface{})

// provide the Iterable interface
func (v iterFunc) Iter() <-chan interface{} {
	ch := make(chan interface{})
	go v(ch)
	return ch
}

// Take returns an Iterable that contains the first n elements of iter.
func Take(iter Iterable, n int) Iterable { return Slice(iter, 0, n) }

// TakeWhile returns an Iterable that contains elements from iter while f is true.
func TakeWhile(iter Iterable, f func(interface{}) bool) Iterable {
	return iterFunc(func(ch chan<- interface{}) {
		for v := range iter.Iter() {
			if !f(v) {
				break
			}
			ch <- v
		}
		close(ch)
	})
}

// Drop returns an Iterable that returns each element of iter after the first n elements.
func Drop(iter Iterable, n int) Iterable {
	return iterFunc(func(ch chan<- interface{}) {
		m := n
		for v := range iter.Iter() {
			if m > 0 {
				m--
				continue
			}
			ch <- v
		}
		close(ch)
	})
}

// DropWhile returns an Iterable that returns each element of iter after the initial sequence for which f returns true.
func DropWhile(iter Iterable, f func(interface{}) bool) Iterable {
	return iterFunc(func(ch chan<- interface{}) {
		drop := true
		for v := range iter.Iter() {
			if drop {
				if f(v) {
					continue
				}
				drop = false
			}
			ch <- v
		}
		close(ch)
	})
}

// Cycle repeats the values of iter in order infinitely.
func Cycle(iter Iterable) Iterable {
	return iterFunc(func(ch chan<- interface{}) {
		for {
			for v := range iter.Iter() {
				ch <- v
			}
		}
	})
}

// Chain returns an Iterable that concatentates all values from the specified Iterables.
func Chain(args []Iterable) Iterable {
	return iterFunc(func(ch chan<- interface{}) {
		for _, e := range args {
			for v := range e.Iter() {
				ch <- v
			}
		}
		close(ch)
	})
}

// Zip returns an Iterable of []interface{} consisting of the next element from
// each input Iterable.  The length of the returned Iterable is the minimum of
// the lengths of the input Iterables.
func Zip(args []Iterable) Iterable {
	return iterFunc(func(ch chan<- interface{}) {
		defer close(ch)
		if len(args) == 0 {
			return
		}
		iters := make([]<-chan interface{}, len(args))
		for i := 0; i < len(iters); i++ {
			iters[i] = args[i].Iter()
		}
		for {
			out := make([]interface{}, len(args))
			for i, v := range iters {
				out[i] = <-v
				if closed(v) {
					return
				}
			}
			ch <- out
		}
	})
}

// ZipWith returns an Iterable containing the result of executing f using arguments read from a and b.
func ZipWith2(f func(c, d interface{}) interface{}, a, b Iterable) Iterable {
	return Map(Zip([]Iterable{a, b}), func(a1 interface{}) interface{} {
		arr := a1.([]interface{})
		return f(arr[0], arr[1])
	})
}

// ZipWith returns an Iterable containing the result of executing f using arguments read from a, b and c.
func ZipWith3(f func(d, e, f interface{}) interface{}, a, b, c Iterable) Iterable {
	return Map(Zip([]Iterable{a, b, c}), func(a1 interface{}) interface{} {
		arr := a1.([]interface{})
		return f(arr[0], arr[1], arr[2])
	})
}

// Slice returns an Iterable that contains the elements from iter
// with indexes in [start, stop).
func Slice(iter Iterable, start, stop int) Iterable {
	return iterFunc(func(ch chan<- interface{}) {
		defer close(ch)
		i := 0
		for v := range iter.Iter() {
			switch {
			case i >= stop:
				return
			case i >= start:
				ch <- v
			}
			i++
		}
	})
}

// Repeat generates an infinite stream of v.
func Repeat(v interface{}) Iterable {
	return iterFunc(func(ch chan<- interface{}) {
		for {
			ch <- v
		}
	})
}

// RepeatTimes generates a stream of n copies of v.
func RepeatTimes(v interface{}, n int) Iterable {
	return iterFunc(func(ch chan<- interface{}) {
		for i := 0; i < n; i++ {
			ch <- v
		}
		close(ch)
	})
}

// Group is the type for elements returned by the GroupBy function.
type Group struct {
	Key  interface{} // key value for matching items
	Vals Iterable    // Iterable for receiving values in the group
}

// Key defines the interface required by the GroupBy function.
type Grouper interface {
	// Return the key for the given value
	Key(interface{}) interface{}

	// Compute equality for the given keys
	Equal(a, b interface{}) bool
}

// GroupBy combines sequences of logically identical values from iter using k
// to generate a key to compare values.  Each value emitted by the returned
// Iterable is of type Group, which contains the key used for matching the
// values for the group, and an Iterable for retrieving all the values in the
// group.
func GroupBy(iter Iterable, k Grouper) Iterable {
	return iterFunc(func(ch chan<- interface{}) {
		var curkey interface{}
		var lst *list.List
		// Basic strategy is to read one group at a time into a list prior to emitting the Group value
		for v := range iter.Iter() {
			kv := k.Key(v)
			if lst == nil || !k.Equal(curkey, kv) {
				if lst != nil {
					ch <- Group{curkey, lst}
				}
				lst = list.New()
				curkey = kv
			}
			lst.PushBack(v)
		}
		if lst != nil {
			ch <- Group{curkey, lst}
		}
		close(ch)
	})
}

// Unique removes duplicate values which occur consecutively using id to compute keys.
func Unique(iter Iterable, id Grouper) Iterable {
	return Map(GroupBy(iter, id), func(v interface{}) interface{} { return v.(Group).Key })
}

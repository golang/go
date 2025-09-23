// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package iter provides basic definitions and operations related to
iterators over sequences.

# Iterators

An iterator is a function that passes successive elements of a
sequence to a callback function, conventionally named yield.
The function stops either when the sequence is finished or
when yield returns false, indicating to stop the iteration early.
This package defines [Seq] and [Seq2]
(pronounced like seek—the first syllable of sequence)
as shorthands for iterators that pass 1 or 2 values per sequence element
to yield:

	type (
		Seq[V any]     func(yield func(V) bool)
		Seq2[K, V any] func(yield func(K, V) bool)
	)

Seq2 represents a sequence of paired values, conventionally key-value
or index-value pairs.

Yield returns true if the iterator should continue with the next
element in the sequence, false if it should stop.

Yield panics if called after it returns false.

For instance, [maps.Keys] returns an iterator that produces the sequence
of keys of the map m, implemented as follows:

	func Keys[Map ~map[K]V, K comparable, V any](m Map) iter.Seq[K] {
		return func(yield func(K) bool) {
			for k := range m {
				if !yield(k) {
					return
				}
			}
		}
	}

Further examples can be found in [The Go Blog: Range Over Function Types].

Iterator functions are most often called by a [range loop], as in:

	func PrintAll[V any](seq iter.Seq[V]) {
		for v := range seq {
			fmt.Println(v)
		}
	}

# Naming Conventions

Iterator functions and methods are named for the sequence being walked:

	// All returns an iterator over all elements in s.
	func (s *Set[V]) All() iter.Seq[V]

The iterator method on a collection type is conventionally named All,
because it iterates a sequence of all the values in the collection.

For a type containing multiple possible sequences, the iterator's name
can indicate which sequence is being provided:

	// Cities returns an iterator over the major cities in the country.
	func (c *Country) Cities() iter.Seq[*City]

	// Languages returns an iterator over the official spoken languages of the country.
	func (c *Country) Languages() iter.Seq[string]

If an iterator requires additional configuration, the constructor function
can take additional configuration arguments:

	// Scan returns an iterator over key-value pairs with min ≤ key ≤ max.
	func (m *Map[K, V]) Scan(min, max K) iter.Seq2[K, V]

	// Split returns an iterator over the (possibly-empty) substrings of s
	// separated by sep.
	func Split(s, sep string) iter.Seq[string]

When there are multiple possible iteration orders, the method name may
indicate that order:

	// All returns an iterator over the list from head to tail.
	func (l *List[V]) All() iter.Seq[V]

	// Backward returns an iterator over the list from tail to head.
	func (l *List[V]) Backward() iter.Seq[V]

	// Preorder returns an iterator over all nodes of the syntax tree
	// beneath (and including) the specified root, in depth-first preorder,
	// visiting a parent node before its children.
	func Preorder(root Node) iter.Seq[Node]

# Single-Use Iterators

Most iterators provide the ability to walk an entire sequence:
when called, the iterator does any setup necessary to start the
sequence, then calls yield on successive elements of the sequence,
and then cleans up before returning. Calling the iterator again
walks the sequence again.

Some iterators break that convention, providing the ability to walk a
sequence only once. These “single-use iterators” typically report values
from a data stream that cannot be rewound to start over.
Calling the iterator again after stopping early may continue the
stream, but calling it again after the sequence is finished will yield
no values at all. Doc comments for functions or methods that return
single-use iterators should document this fact:

	// Lines returns an iterator over lines read from r.
	// It returns a single-use iterator.
	func (r *Reader) Lines() iter.Seq[string]

# Pulling Values

Functions and methods that accept or return iterators
should use the standard [Seq] or [Seq2] types, to ensure
compatibility with range loops and other iterator adapters.
The standard iterators can be thought of as “push iterators”, which
push values to the yield function.

Sometimes a range loop is not the most natural way to consume values
of the sequence. In this case, [Pull] converts a standard push iterator
to a “pull iterator”, which can be called to pull one value at a time
from the sequence. [Pull] starts an iterator and returns a pair
of functions—next and stop—which return the next value from the iterator
and stop it, respectively.

For example:

	// Pairs returns an iterator over successive pairs of values from seq.
	func Pairs[V any](seq iter.Seq[V]) iter.Seq2[V, V] {
		return func(yield func(V, V) bool) {
			next, stop := iter.Pull(seq)
			defer stop()
			for {
				v1, ok1 := next()
				if !ok1 {
					return
				}
				v2, ok2 := next()
				// If ok2 is false, v2 should be the
				// zero value; yield one last pair.
				if !yield(v1, v2) {
					return
				}
				if !ok2 {
					return
				}
			}
		}
	}

If clients do not consume the sequence to completion, they must call stop,
which allows the iterator function to finish and return. As shown in
the example, the conventional way to ensure this is to use defer.

# Standard Library Usage

A few packages in the standard library provide iterator-based APIs,
most notably the [maps] and [slices] packages.
For example, [maps.Keys] returns an iterator over the keys of a map,
while [slices.Sorted] collects the values of an iterator into a slice,
sorts them, and returns the slice, so to iterate over the sorted keys of a map:

	for _, key := range slices.Sorted(maps.Keys(m)) {
		...
	}

# Mutation

Iterators provide only the values of the sequence, not any direct way
to modify it. If an iterator wishes to provide a mechanism for modifying
a sequence during iteration, the usual approach is to define a position type
with the extra operations and then provide an iterator over positions.

For example, a tree implementation might provide:

	// Positions returns an iterator over positions in the sequence.
	func (t *Tree[V]) Positions() iter.Seq[*Pos[V]]

	// A Pos represents a position in the sequence.
	// It is only valid during the yield call it is passed to.
	type Pos[V any] struct { ... }

	// Pos returns the value at the cursor.
	func (p *Pos[V]) Value() V

	// Delete deletes the value at this point in the iteration.
	func (p *Pos[V]) Delete()

	// Set changes the value v at the cursor.
	func (p *Pos[V]) Set(v V)

And then a client could delete boring values from the tree using:

	for p := range t.Positions() {
		if boring(p.Value()) {
			p.Delete()
		}
	}

[The Go Blog: Range Over Function Types]: https://go.dev/blog/range-functions
[range loop]: https://go.dev/ref/spec#For_range
*/
package iter

import (
	"internal/race"
	"runtime"
	"unsafe"
)

// Seq is an iterator over sequences of individual values.
// When called as seq(yield), seq calls yield(v) for each value v in the sequence,
// stopping early if yield returns false.
// See the [iter] package documentation for more details.
type Seq[V any] func(yield func(V) bool)

// Seq2 is an iterator over sequences of pairs of values, most commonly key-value pairs.
// When called as seq(yield), seq calls yield(k, v) for each pair (k, v) in the sequence,
// stopping early if yield returns false.
// See the [iter] package documentation for more details.
type Seq2[K, V any] func(yield func(K, V) bool)

type coro struct{}

//go:linkname newcoro runtime.newcoro
func newcoro(func(*coro)) *coro

//go:linkname coroswitch runtime.coroswitch
func coroswitch(*coro)

// Pull converts the “push-style” iterator sequence seq
// into a “pull-style” iterator accessed by the two functions
// next and stop.
//
// Next returns the next value in the sequence
// and a boolean indicating whether the value is valid.
// When the sequence is over, next returns the zero V and false.
// It is valid to call next after reaching the end of the sequence
// or after calling stop. These calls will continue
// to return the zero V and false.
//
// Stop ends the iteration. It must be called when the caller is
// no longer interested in next values and next has not yet
// signaled that the sequence is over (with a false boolean return).
// It is valid to call stop multiple times and when next has
// already returned false. Typically, callers should “defer stop()”.
//
// It is an error to call next or stop from multiple goroutines
// simultaneously.
//
// If the iterator panics during a call to next (or stop),
// then next (or stop) itself panics with the same value.
func Pull[V any](seq Seq[V]) (next func() (V, bool), stop func()) {
	var pull struct {
		v          V
		ok         bool
		done       bool
		yieldNext  bool
		seqDone    bool // to detect Goexit
		racer      int
		panicValue any
	}
	c := newcoro(func(c *coro) {
		race.Acquire(unsafe.Pointer(&pull.racer))
		if pull.done {
			race.Release(unsafe.Pointer(&pull.racer))
			return
		}
		yield := func(v1 V) bool {
			if pull.done {
				return false
			}
			if !pull.yieldNext {
				panic("iter.Pull: yield called again before next")
			}
			pull.yieldNext = false
			pull.v, pull.ok = v1, true
			race.Release(unsafe.Pointer(&pull.racer))
			coroswitch(c)
			race.Acquire(unsafe.Pointer(&pull.racer))
			return !pull.done
		}
		// Recover and propagate panics from seq.
		defer func() {
			if p := recover(); p != nil {
				pull.panicValue = p
			} else if !pull.seqDone {
				pull.panicValue = goexitPanicValue
			}
			pull.done = true // Invalidate iterator
			race.Release(unsafe.Pointer(&pull.racer))
		}()
		seq(yield)
		var v0 V
		pull.v, pull.ok = v0, false
		pull.seqDone = true
	})
	next = func() (v1 V, ok1 bool) {
		race.Write(unsafe.Pointer(&pull.racer)) // detect races

		if pull.done {
			return
		}
		if pull.yieldNext {
			panic("iter.Pull: next called again before yield")
		}
		pull.yieldNext = true
		race.Release(unsafe.Pointer(&pull.racer))
		coroswitch(c)
		race.Acquire(unsafe.Pointer(&pull.racer))

		// Propagate panics and goexits from seq.
		if pull.panicValue != nil {
			if pull.panicValue == goexitPanicValue {
				// Propagate runtime.Goexit from seq.
				runtime.Goexit()
			} else {
				panic(pull.panicValue)
			}
		}
		return pull.v, pull.ok
	}
	stop = func() {
		race.Write(unsafe.Pointer(&pull.racer)) // detect races

		if !pull.done {
			pull.done = true
			race.Release(unsafe.Pointer(&pull.racer))
			coroswitch(c)
			race.Acquire(unsafe.Pointer(&pull.racer))

			// Propagate panics and goexits from seq.
			if pull.panicValue != nil {
				if pull.panicValue == goexitPanicValue {
					// Propagate runtime.Goexit from seq.
					runtime.Goexit()
				} else {
					panic(pull.panicValue)
				}
			}
		}
	}
	return next, stop
}

// Pull2 converts the “push-style” iterator sequence seq
// into a “pull-style” iterator accessed by the two functions
// next and stop.
//
// Next returns the next pair in the sequence
// and a boolean indicating whether the pair is valid.
// When the sequence is over, next returns a pair of zero values and false.
// It is valid to call next after reaching the end of the sequence
// or after calling stop. These calls will continue
// to return a pair of zero values and false.
//
// Stop ends the iteration. It must be called when the caller is
// no longer interested in next values and next has not yet
// signaled that the sequence is over (with a false boolean return).
// It is valid to call stop multiple times and when next has
// already returned false. Typically, callers should “defer stop()”.
//
// It is an error to call next or stop from multiple goroutines
// simultaneously.
//
// If the iterator panics during a call to next (or stop),
// then next (or stop) itself panics with the same value.
func Pull2[K, V any](seq Seq2[K, V]) (next func() (K, V, bool), stop func()) {
	var pull struct {
		k          K
		v          V
		ok         bool
		done       bool
		yieldNext  bool
		seqDone    bool
		racer      int
		panicValue any
	}
	c := newcoro(func(c *coro) {
		race.Acquire(unsafe.Pointer(&pull.racer))
		if pull.done {
			race.Release(unsafe.Pointer(&pull.racer))
			return
		}
		yield := func(k1 K, v1 V) bool {
			if pull.done {
				return false
			}
			if !pull.yieldNext {
				panic("iter.Pull2: yield called again before next")
			}
			pull.yieldNext = false
			pull.k, pull.v, pull.ok = k1, v1, true
			race.Release(unsafe.Pointer(&pull.racer))
			coroswitch(c)
			race.Acquire(unsafe.Pointer(&pull.racer))
			return !pull.done
		}
		// Recover and propagate panics from seq.
		defer func() {
			if p := recover(); p != nil {
				pull.panicValue = p
			} else if !pull.seqDone {
				pull.panicValue = goexitPanicValue
			}
			pull.done = true // Invalidate iterator.
			race.Release(unsafe.Pointer(&pull.racer))
		}()
		seq(yield)
		var k0 K
		var v0 V
		pull.k, pull.v, pull.ok = k0, v0, false
		pull.seqDone = true
	})
	next = func() (k1 K, v1 V, ok1 bool) {
		race.Write(unsafe.Pointer(&pull.racer)) // detect races

		if pull.done {
			return
		}
		if pull.yieldNext {
			panic("iter.Pull2: next called again before yield")
		}
		pull.yieldNext = true
		race.Release(unsafe.Pointer(&pull.racer))
		coroswitch(c)
		race.Acquire(unsafe.Pointer(&pull.racer))

		// Propagate panics and goexits from seq.
		if pull.panicValue != nil {
			if pull.panicValue == goexitPanicValue {
				// Propagate runtime.Goexit from seq.
				runtime.Goexit()
			} else {
				panic(pull.panicValue)
			}
		}
		return pull.k, pull.v, pull.ok
	}
	stop = func() {
		race.Write(unsafe.Pointer(&pull.racer)) // detect races

		if !pull.done {
			pull.done = true
			race.Release(unsafe.Pointer(&pull.racer))
			coroswitch(c)
			race.Acquire(unsafe.Pointer(&pull.racer))

			// Propagate panics and goexits from seq.
			if pull.panicValue != nil {
				if pull.panicValue == goexitPanicValue {
					// Propagate runtime.Goexit from seq.
					runtime.Goexit()
				} else {
					panic(pull.panicValue)
				}
			}
		}
	}
	return next, stop
}

// goexitPanicValue is a sentinel value indicating that an iterator
// exited via runtime.Goexit.
var goexitPanicValue any = new(int)

// run

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package orderedmap provides an ordered map, implemented as a binary tree.
package main

import (
	"bytes"
	"context"
	"fmt"
	"runtime"
)

type Ordered interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 |
		~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 | ~uintptr |
		~float32 | ~float64 |
		~string
}

// _Map is an ordered map.
type _Map[K, V any] struct {
	root    *node[K, V]
	compare func(K, K) int
}

// node is the type of a node in the binary tree.
type node[K, V any] struct {
	key         K
	val         V
	left, right *node[K, V]
}

// _New returns a new map. It takes a comparison function that compares two
// keys and returns < 0 if the first is less, == 0 if they are equal,
// > 0 if the first is greater.
func _New[K, V any](compare func(K, K) int) *_Map[K, V] {
	return &_Map[K, V]{compare: compare}
}

// _NewOrdered returns a new map whose key is an ordered type.
// This is like _New, but does not require providing a compare function.
// The map compare function uses the obvious key ordering.
func _NewOrdered[K Ordered, V any]() *_Map[K, V] {
	return _New[K, V](func(k1, k2 K) int {
		switch {
		case k1 < k2:
			return -1
		case k1 == k2:
			return 0
		default:
			return 1
		}
	})
}

// find looks up key in the map, returning either a pointer to the slot of the
// node holding key, or a pointer to the slot where should a node would go.
func (m *_Map[K, V]) find(key K) **node[K, V] {
	pn := &m.root
	for *pn != nil {
		switch cmp := m.compare(key, (*pn).key); {
		case cmp < 0:
			pn = &(*pn).left
		case cmp > 0:
			pn = &(*pn).right
		default:
			return pn
		}
	}
	return pn
}

// Insert inserts a new key/value into the map.
// If the key is already present, the value is replaced.
// Reports whether this is a new key.
func (m *_Map[K, V]) Insert(key K, val V) bool {
	pn := m.find(key)
	if *pn != nil {
		(*pn).val = val
		return false
	}
	*pn = &node[K, V]{key: key, val: val}
	return true
}

// Find returns the value associated with a key, or the zero value
// if not present. The found result reports whether the key was found.
func (m *_Map[K, V]) Find(key K) (V, bool) {
	pn := m.find(key)
	if *pn == nil {
		var zero V
		return zero, false
	}
	return (*pn).val, true
}

// keyValue is a pair of key and value used while iterating.
type keyValue[K, V any] struct {
	key K
	val V
}

// iterate returns an iterator that traverses the map.
func (m *_Map[K, V]) Iterate() *_Iterator[K, V] {
	sender, receiver := _Ranger[keyValue[K, V]]()
	var f func(*node[K, V]) bool
	f = func(n *node[K, V]) bool {
		if n == nil {
			return true
		}
		// Stop the traversal if Send fails, which means that
		// nothing is listening to the receiver.
		return f(n.left) &&
			sender.Send(context.Background(), keyValue[K, V]{n.key, n.val}) &&
			f(n.right)
	}
	go func() {
		f(m.root)
		sender.Close()
	}()
	return &_Iterator[K, V]{receiver}
}

// _Iterator is used to iterate over the map.
type _Iterator[K, V any] struct {
	r *_Receiver[keyValue[K, V]]
}

// Next returns the next key and value pair, and a boolean that reports
// whether they are valid. If not valid, we have reached the end of the map.
func (it *_Iterator[K, V]) Next() (K, V, bool) {
	keyval, ok := it.r.Next(context.Background())
	if !ok {
		var zerok K
		var zerov V
		return zerok, zerov, false
	}
	return keyval.key, keyval.val, true
}

func TestMap() {
	m := _New[[]byte, int](bytes.Compare)

	if _, found := m.Find([]byte("a")); found {
		panic(fmt.Sprintf("unexpectedly found %q in empty map", []byte("a")))
	}
	if !m.Insert([]byte("a"), 'a') {
		panic(fmt.Sprintf("key %q unexpectedly already present", []byte("a")))
	}
	if !m.Insert([]byte("c"), 'c') {
		panic(fmt.Sprintf("key %q unexpectedly already present", []byte("c")))
	}
	if !m.Insert([]byte("b"), 'b') {
		panic(fmt.Sprintf("key %q unexpectedly already present", []byte("b")))
	}
	if m.Insert([]byte("c"), 'x') {
		panic(fmt.Sprintf("key %q unexpectedly not present", []byte("c")))
	}

	if v, found := m.Find([]byte("a")); !found {
		panic(fmt.Sprintf("did not find %q", []byte("a")))
	} else if v != 'a' {
		panic(fmt.Sprintf("key %q returned wrong value %c, expected %c", []byte("a"), v, 'a'))
	}
	if v, found := m.Find([]byte("c")); !found {
		panic(fmt.Sprintf("did not find %q", []byte("c")))
	} else if v != 'x' {
		panic(fmt.Sprintf("key %q returned wrong value %c, expected %c", []byte("c"), v, 'x'))
	}

	if _, found := m.Find([]byte("d")); found {
		panic(fmt.Sprintf("unexpectedly found %q", []byte("d")))
	}

	gather := func(it *_Iterator[[]byte, int]) []int {
		var r []int
		for {
			_, v, ok := it.Next()
			if !ok {
				return r
			}
			r = append(r, v)
		}
	}
	got := gather(m.Iterate())
	want := []int{'a', 'b', 'x'}
	if !_SliceEqual(got, want) {
		panic(fmt.Sprintf("Iterate returned %v, want %v", got, want))
	}
}

func main() {
	TestMap()
}

// _Equal reports whether two slices are equal: the same length and all
// elements equal. All floating point NaNs are considered equal.
func _SliceEqual[Elem comparable](s1, s2 []Elem) bool {
	if len(s1) != len(s2) {
		return false
	}
	for i, v1 := range s1 {
		v2 := s2[i]
		if v1 != v2 {
			isNaN := func(f Elem) bool { return f != f }
			if !isNaN(v1) || !isNaN(v2) {
				return false
			}
		}
	}
	return true
}

// Ranger returns a Sender and a Receiver. The Receiver provides a
// Next method to retrieve values. The Sender provides a Send method
// to send values and a Close method to stop sending values. The Next
// method indicates when the Sender has been closed, and the Send
// method indicates when the Receiver has been freed.
//
// This is a convenient way to exit a goroutine sending values when
// the receiver stops reading them.
func _Ranger[Elem any]() (*_Sender[Elem], *_Receiver[Elem]) {
	c := make(chan Elem)
	d := make(chan struct{})
	s := &_Sender[Elem]{
		values: c,
		done:   d,
	}
	r := &_Receiver[Elem]{
		values: c,
		done:   d,
	}
	runtime.SetFinalizer(r, (*_Receiver[Elem]).finalize)
	return s, r
}

// A _Sender is used to send values to a Receiver.
type _Sender[Elem any] struct {
	values chan<- Elem
	done   <-chan struct{}
}

// Send sends a value to the receiver. It reports whether the value was sent.
// The value will not be sent if the context is closed or the receiver
// is freed.
func (s *_Sender[Elem]) Send(ctx context.Context, v Elem) bool {
	select {
	case <-ctx.Done():
		return false
	case s.values <- v:
		return true
	case <-s.done:
		return false
	}
}

// Close tells the receiver that no more values will arrive.
// After Close is called, the _Sender may no longer be used.
func (s *_Sender[Elem]) Close() {
	close(s.values)
}

// A _Receiver receives values from a _Sender.
type _Receiver[Elem any] struct {
	values <-chan Elem
	done   chan<- struct{}
}

// Next returns the next value from the channel. The bool result indicates
// whether the value is valid.
func (r *_Receiver[Elem]) Next(ctx context.Context) (v Elem, ok bool) {
	select {
	case <-ctx.Done():
	case v, ok = <-r.values:
	}
	return v, ok
}

// finalize is a finalizer for the receiver.
func (r *_Receiver[Elem]) finalize() {
	close(r.done)
}

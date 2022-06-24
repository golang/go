// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

import (
	"context"
	"runtime"
)

type Ordered interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 |
		~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 | ~uintptr |
		~float32 | ~float64 |
		~string
}

// Map is an ordered map.
type Map[K, V any] struct {
	root    *node[K, V]
	compare func(K, K) int
}

// node is the type of a node in the binary tree.
type node[K, V any] struct {
	key         K
	val         V
	left, right *node[K, V]
}

// New returns a new map. It takes a comparison function that compares two
// keys and returns < 0 if the first is less, == 0 if they are equal,
// > 0 if the first is greater.
func New[K, V any](compare func(K, K) int) *Map[K, V] {
	return &Map[K, V]{compare: compare}
}

// NewOrdered returns a new map whose key is an ordered type.
// This is like New, but does not require providing a compare function.
// The map compare function uses the obvious key ordering.
func NewOrdered[K Ordered, V any]() *Map[K, V] {
	return New[K, V](func(k1, k2 K) int {
		switch {
		case k1 < k2:
			return -1
		case k1 > k2:
			return 1
		default:
			return 0
		}
	})
}

// find looks up key in the map, returning either a pointer to the slot of the
// node holding key, or a pointer to the slot where a node would go.
func (m *Map[K, V]) find(key K) **node[K, V] {
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
func (m *Map[K, V]) Insert(key K, val V) bool {
	pn := m.find(key)
	if *pn != nil {
		(*pn).val = val
		return false
	}
	*pn = &node[K, V]{key: key, val: val}
	return true
}

// Find returns the value associated with a key, or the zero value
// if not present. The second result reports whether the key was found.
func (m *Map[K, V]) Find(key K) (V, bool) {
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
func (m *Map[K, V]) Iterate() *Iterator[K, V] {
	sender, receiver := Ranger[keyValue[K, V]]()
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
	return &Iterator[K, V]{receiver}
}

// Iterator is used to iterate over the map.
type Iterator[K, V any] struct {
	r *Receiver[keyValue[K, V]]
}

// Next returns the next key and value pair, and a boolean that reports
// whether they are valid. If not valid, we have reached the end of the map.
func (it *Iterator[K, V]) Next() (K, V, bool) {
	keyval, ok := it.r.Next(context.Background())
	if !ok {
		var zerok K
		var zerov V
		return zerok, zerov, false
	}
	return keyval.key, keyval.val, true
}

// Equal reports whether two slices are equal: the same length and all
// elements equal. All floating point NaNs are considered equal.
func SliceEqual[Elem comparable](s1, s2 []Elem) bool {
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
func Ranger[Elem any]() (*Sender[Elem], *Receiver[Elem]) {
	c := make(chan Elem)
	d := make(chan struct{})
	s := &Sender[Elem]{
		values: c,
		done:   d,
	}
	r := &Receiver[Elem]{
		values: c,
		done:   d,
	}
	runtime.SetFinalizer(r, (*Receiver[Elem]).finalize)
	return s, r
}

// A Sender is used to send values to a Receiver.
type Sender[Elem any] struct {
	values chan<- Elem
	done   <-chan struct{}
}

// Send sends a value to the receiver. It reports whether the value was sent.
// The value will not be sent if the context is closed or the receiver
// is freed.
func (s *Sender[Elem]) Send(ctx context.Context, v Elem) bool {
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
// After Close is called, the Sender may no longer be used.
func (s *Sender[Elem]) Close() {
	close(s.values)
}

// A Receiver receives values from a Sender.
type Receiver[Elem any] struct {
	values <-chan Elem
	done   chan<- struct{}
}

// Next returns the next value from the channel. The bool result indicates
// whether the value is valid.
func (r *Receiver[Elem]) Next(ctx context.Context) (v Elem, ok bool) {
	select {
	case <-ctx.Done():
	case v, ok = <-r.values:
	}
	return v, ok
}

// finalize is a finalizer for the receiver.
func (r *Receiver[Elem]) finalize() {
	close(r.done)
}

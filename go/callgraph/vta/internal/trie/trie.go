// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// trie implements persistent Patricia trie maps.
//
// Each Map is effectively a map from uint64 to interface{}. Patricia tries are
// a form of radix tree that are particularly appropriate when many maps will be
// created, merged together and large amounts of sharing are expected (e.g.
// environment abstract domains in program analysis).
//
// This implementation closely follows the paper:
//   C. Okasaki and A. Gill, “Fast mergeable integer maps,” in ACM SIGPLAN
//   Workshop on ML, September 1998, pp. 77–86.
// Each Map is immutable and can be read from concurrently. The map does not
// guarantee that the value pointed to by the interface{} value is not updated
// concurrently.
//
// These Maps are optimized for situations where there will be many maps created at
// with a high degree of sharing and combining of maps together. If you do not expect,
// significant amount of sharing, the builtin map[T]U is much better choice!
//
// Each Map is created by a Builder. Each Builder has a unique Scope and each node is
// created within this scope. Maps x and y are == if they contains the same
// (key,value) mappings and have equal scopes.
//
// Internally these are big endian Patricia trie nodes, and the keys are sorted.
package trie

import (
	"fmt"
	"strings"
)

// Map is effectively a finite mapping from uint64 keys to interface{} values.
// Maps are immutable and can be read from concurrently.
//
// Notes on concurrency:
// - A Map value itself is an interface and assignments to a Map value can race.
// - Map does not guarantee that the value pointed to by the interface{} value
//   is not updated concurrently.
type Map struct {
	s Scope
	n node
}

func (m Map) Scope() Scope {
	return m.s
}
func (m Map) Size() int {
	if m.n == nil {
		return 0
	}
	return m.n.size()
}
func (m Map) Lookup(k uint64) (interface{}, bool) {
	if m.n != nil {
		if leaf := m.n.find(key(k)); leaf != nil {
			return leaf.v, true
		}
	}
	return nil, false
}

// Converts the map into a {<key>: <value>[, ...]} string. This uses the default
// %s string conversion for <value>.
func (m Map) String() string {
	var kvs []string
	m.Range(func(u uint64, i interface{}) bool {
		kvs = append(kvs, fmt.Sprintf("%d: %s", u, i))
		return true
	})
	return fmt.Sprintf("{%s}", strings.Join(kvs, ", "))
}

// Range over the leaf (key, value) pairs in the map in order and
// applies cb(key, value) to each. Stops early if cb returns false.
// Returns true if all elements were visited without stopping early.
func (m Map) Range(cb func(uint64, interface{}) bool) bool {
	if m.n != nil {
		return m.n.visit(cb)
	}
	return true
}

// DeepEqual returns true if m and other contain the same (k, v) mappings
// [regardless of Scope].
//
// Equivalently m.DeepEqual(other) <=> reflect.DeepEqual(Elems(m), Elems(other))
func (m Map) DeepEqual(other Map) bool {
	if m.Scope() == other.Scope() {
		return m.n == other.n
	}
	if (m.n == nil) || (other.n == nil) {
		return m.Size() == 0 && other.Size() == 0
	}
	return m.n.deepEqual(other.n)
}

// Elems are the (k,v) elements in the Map as a map[uint64]interface{}
func Elems(m Map) map[uint64]interface{} {
	dest := make(map[uint64]interface{}, m.Size())
	m.Range(func(k uint64, v interface{}) bool {
		dest[k] = v
		return true
	})
	return dest
}

// node is an internal node within a trie map.
// A node is either empty, a leaf or a branch.
type node interface {
	size() int

	// visit the leaves (key, value) pairs in the map in order and
	// applies cb(key, value) to each. Stops early if cb returns false.
	// Returns true if all elements were visited without stopping early.
	visit(cb func(uint64, interface{}) bool) bool

	// Two nodes contain the same elements regardless of scope.
	deepEqual(node) bool

	// find the leaf for the given key value or nil if it is not present.
	find(k key) *leaf

	// implementations must implement this.
	nodeImpl()
}

// empty represents the empty map within a scope.
//
// The current builder ensure
type empty struct {
	s Scope
}

// leaf represents a single <key, value> pair.
type leaf struct {
	k key
	v interface{}
}

// branch represents a tree node within the Patricia trie.
//
// All keys within the branch match a `prefix` of the key
// up to a `branching` bit, and the left and right nodes
// contain keys that disagree on the bit at the `branching` bit.
type branch struct {
	sz        int    // size. cached for O(1) lookup
	prefix    prefix // == mask(p0, branching) for some p0
	branching bitpos

	// Invariants:
	// - neither is nil.
	// - neither is *empty.
	// - all keys in left are <= p.
	// - all keys in right are > p.
	left, right node
}

// all of these types are Maps.
var _ node = &empty{}
var _ node = &leaf{}
var _ node = &branch{}

func (*empty) nodeImpl()  {}
func (*leaf) nodeImpl()   {}
func (*branch) nodeImpl() {}

func (*empty) find(k key) *leaf { return nil }
func (l *leaf) find(k key) *leaf {
	if k == l.k {
		return l
	}
	return nil
}
func (br *branch) find(k key) *leaf {
	kp := prefix(k)
	if !matchPrefix(kp, br.prefix, br.branching) {
		return nil
	}
	if zeroBit(kp, br.branching) {
		return br.left.find(k)
	}
	return br.right.find(k)
}

func (*empty) size() int     { return 0 }
func (*leaf) size() int      { return 1 }
func (br *branch) size() int { return br.sz }

func (*empty) deepEqual(m node) bool {
	_, ok := m.(*empty)
	return ok
}
func (l *leaf) deepEqual(m node) bool {
	if m, ok := m.(*leaf); ok {
		return m == l || (l.k == m.k && l.v == m.v)
	}
	return false
}

func (br *branch) deepEqual(m node) bool {
	if m, ok := m.(*branch); ok {
		if br == m {
			return true
		}
		return br.sz == m.sz && br.branching == m.branching && br.prefix == m.prefix &&
			br.left.deepEqual(m.left) && br.right.deepEqual(m.right)
	}
	// if m is not a branch, m contains 0 or 1 elem.
	// br contains at least 2 keys that disagree on a prefix.
	return false
}

func (*empty) visit(cb func(uint64, interface{}) bool) bool {
	return true
}
func (l *leaf) visit(cb func(uint64, interface{}) bool) bool {
	return cb(uint64(l.k), l.v)
}
func (br *branch) visit(cb func(uint64, interface{}) bool) bool {
	if !br.left.visit(cb) {
		return false
	}
	return br.right.visit(cb)
}

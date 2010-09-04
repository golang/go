// $G $F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// ----------------------------------------------------------------------------
// Helper functions

func ASSERT(p bool) {
	if !p {
		// panic 0
	}
}


// ----------------------------------------------------------------------------
// Implementation of the HashMap

type KeyType interface {
	Hash() uint32
	Match(other *KeyType) bool
}


type ValueType interface {
	// empty interface
}


type Entry struct {
	key *KeyType
	value *ValueType
}


type Array [1024]Entry

type HashMap struct {
	map_ *Array
	log2_capacity_ uint32
	occupancy_ uint32
}


func (m *HashMap) capacity() uint32 {
	return 1 << m.log2_capacity_
}


func (m *HashMap) Clear() {
	// Mark all entries as empty.
	var i uint32 = m.capacity() - 1
	for i > 0 {
		m.map_[i].key = nil
		i = i - 1
	}
	m.occupancy_ = 0
}


func (m *HashMap) Initialize (initial_log2_capacity uint32) {
	m.log2_capacity_ = initial_log2_capacity
	m.map_ = new(Array)
	m.Clear()
}


func (m *HashMap) Probe (key *KeyType) *Entry {
	ASSERT(key != nil)

	var i uint32 = key.Hash() % m.capacity()
	ASSERT(0 <= i && i < m.capacity())

	ASSERT(m.occupancy_ < m.capacity())	// guarantees loop termination
	for m.map_[i].key != nil && !m.map_[i].key.Match(key) {
		i++
		if i >= m.capacity() {
			i = 0
		}
	}

	return &m.map_[i]
}


func (m *HashMap) Lookup (key *KeyType, insert bool) *Entry {
	// Find a matching entry.
	var p *Entry = m.Probe(key)
		if p.key != nil {
		return p
	}

	// No entry found; insert one if necessary.
	if insert {
		p.key = key
		p.value = nil
		m.occupancy_++

		// Grow the map if we reached >= 80% occupancy.
		if m.occupancy_ + m.occupancy_/4 >= m.capacity() {
			m.Resize()
			p = m.Probe(key)
		}

		return p
	}

	// No entry found and none inserted.
	return nil
}


func (m *HashMap) Resize() {
	var hmap *Array = m.map_
	var n uint32 = m.occupancy_

	// Allocate a new map of twice the current size.
	m.Initialize(m.log2_capacity_ << 1)

	// Rehash all current entries.
	var i uint32 = 0
	for n > 0 {
		if hmap[i].key != nil {
			m.Lookup(hmap[i].key, true).value = hmap[i].value
			n = n - 1
		}
		i++
	}
}


// ----------------------------------------------------------------------------
// Test code

type Number struct {
	x uint32
}


func (n *Number) Hash() uint32 {
	return n.x * 23
}


func (n *Number) Match(other *KeyType) bool {
	// var y *Number = other
	// return n.x == y.x
	return false
}


func MakeNumber (x uint32) *Number {
	var n *Number = new(Number)
	n.x = x
	return n
}


func main() {
	// func (n int) int { return n + 1; }(1)

	//print "HashMap - gri 2/8/2008\n"

	var hmap *HashMap = new(HashMap)
	hmap.Initialize(0)

	var x1 *Number = MakeNumber(1001)
	var x2 *Number = MakeNumber(2002)
	var x3 *Number = MakeNumber(3003)
	_, _, _ = x1, x2, x3

	// this doesn't work I think...
	//hmap.Lookup(x1, true)
	//hmap.Lookup(x2, true)
	//hmap.Lookup(x3, true)

	//print "done\n"
}

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssacore

import (
	"fmt"

	"cmd/internal/src"
)

// An XPosMap is a map from fileindex and line of src.XPos to int32,
// implemented sparsely to save space (column and statement status are ignored).
// The sparse skeleton is constructed once, and then reused by ssa phases
// that (re)move values with statements attached.
type XPosMap struct {
	// A map from file index to maps from line range to integers (block numbers)
	maps map[int32]*BiasedSparseMap
	// The next two fields provide a single-item cache for common case of repeated lines from same file.
	lastIndex int32            // -1 means no entry in cache
	lastMap   *BiasedSparseMap // map found at maps[lastIndex]
}

// NewXPosMap constructs an xposmap valid for inputs which have a file index in the keys of x,
// and line numbers in the range x[file index].
// The resulting xposmap will panic if a caller attempts to set or add an XPos not in that range.
func NewXPosMap(x map[int]LineRange) *XPosMap {
	maps := make(map[int32]*BiasedSparseMap)
	for i, p := range x {
		maps[int32(i)] = newBiasedSparseMap(int(p.First), int(p.Last))
	}
	return &XPosMap{maps: maps, lastIndex: -1} // zero for the rest is okay
}

type LineRange struct {
	First, Last uint32
}

// Clear removes data from the map but leaves the sparse skeleton.
func (m *XPosMap) Clear() {
	for _, l := range m.maps {
		if l != nil {
			l.clear()
		}
	}
	m.lastIndex = -1
	m.lastMap = nil
}

// MapFor returns the line range map for a given file index.
func (m *XPosMap) MapFor(index int32) *BiasedSparseMap {
	if index == m.lastIndex {
		return m.lastMap
	}
	mf := m.maps[index]
	m.lastIndex = index
	m.lastMap = mf
	return mf
}

// Set inserts p->v into the map.
// If p does not fall within the Set of fileindex->lineRange used to construct m, this will panic.
func (m *XPosMap) Set(p src.XPos, v int32) {
	s := m.MapFor(p.FileIndex())
	if s == nil {
		panic(fmt.Sprintf("xposmap.set(%d), file index not found in map\n", p.FileIndex()))
	}
	s.set(p.Line(), v)
}

// Get returns the int32 associated with the file index and line of p.
func (m *XPosMap) Get(p src.XPos) (int32, bool) {
	s := m.MapFor(p.FileIndex())
	if s == nil {
		return 0, false
	}
	return s.get(p.Line())
}

// Add adds p to m, treating m as a set instead of as a map.
// If p does not fall within the set of fileindex->lineRange used to construct m, this will panic.
// Use clear() in between set/map interpretations of m.
func (m *XPosMap) Add(p src.XPos) {
	m.Set(p, 0)
}

// Contains returns whether the file index and line of p are in m,
// treating m as a set instead of as a map.
func (m *XPosMap) Contains(p src.XPos) bool {
	s := m.MapFor(p.FileIndex())
	if s == nil {
		return false
	}
	return s.contains(p.Line())
}

// Remove removes the file index and line for p from m,
// whether m is currently treated as a map or set.
func (m *XPosMap) Remove(p src.XPos) {
	s := m.MapFor(p.FileIndex())
	if s == nil {
		return
	}
	s.remove(p.Line())
}

// ForeachEntry applies f to each (fileindex, line, value) triple in m.
func (m *XPosMap) ForeachEntry(f func(j int32, l uint, v int32)) {
	for j, mm := range m.maps {
		s := mm.size()
		for i := 0; i < s; i++ {
			l, v := mm.getEntry(i)
			f(j, l, v)
		}
	}
}

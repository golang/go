// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typerefs

import (
	"fmt"
	"math/bits"
	"sort"
	"strings"
	"sync"

	"golang.org/x/tools/gopls/internal/lsp/source"
)

// PackageIndex stores common data to enable efficient representation of
// references and package sets.
type PackageIndex struct {
	// For now, PackageIndex just indexes package ids, to save space and allow for
	// faster unions via sparse int vectors.
	mu  sync.Mutex
	ids []source.PackageID
	m   map[source.PackageID]int
}

// NewPackageIndex creates a new PackageIndex instance for use in building
// reference and package sets.
func NewPackageIndex() *PackageIndex {
	return &PackageIndex{
		m: make(map[source.PackageID]int),
	}
}

// idx returns the packageIdx referencing id, creating one if id is not yet
// tracked by the receiver.
func (r *PackageIndex) idx(id source.PackageID) int {
	r.mu.Lock()
	defer r.mu.Unlock()
	if i, ok := r.m[id]; ok {
		return i
	}
	i := len(r.ids)
	r.m[id] = i
	r.ids = append(r.ids, id)
	return i
}

// id returns the PackageID for idx.
//
// idx must have been created by this PackageIndex instance.
func (r *PackageIndex) id(idx int) source.PackageID {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.ids[idx]
}

// A PackageSet is a set of source.PackageIDs, optimized for inuse memory
// footprint and efficient union operations.
type PackageSet struct {
	// PackageSet is a sparse int vector of package indexes from parent.
	parent *PackageIndex
	sparse map[int]blockType // high bits in key, set of low bits in value
}

type blockType = uint // type of each sparse vector element
const blockSize = bits.UintSize

// NewSet creates a new PackageSet bound to this PackageIndex instance.
//
// PackageSets may only be combined with other PackageSets from the same
// instance.
func (s *PackageIndex) NewSet() *PackageSet {
	return &PackageSet{
		parent: s,
		sparse: make(map[int]blockType),
	}
}

// Add records a new element in the package set.
func (s *PackageSet) Add(id source.PackageID) {
	s.add(s.parent.idx(id))
}

func (s *PackageSet) add(idx int) {
	i := int(idx)
	s.sparse[i/blockSize] |= 1 << (i % blockSize)
}

// Union records all elements from other into the receiver, mutating the
// receiver set but not the argument set. The receiver must not be nil, but the
// argument set may be nil.
//
// Precondition: both package sets were created with the same PackageIndex.
func (s *PackageSet) Union(other *PackageSet) {
	if other == nil {
		return // e.g. unsafe
	}
	if other.parent != s.parent {
		panic("other set is from a different PackageIndex instance")
	}
	for k, v := range other.sparse {
		if v0 := s.sparse[k]; v0 != v {
			s.sparse[k] = v0 | v
		}
	}
}

// Contains reports whether id is contained in the receiver set.
func (s *PackageSet) Contains(id source.PackageID) bool {
	i := int(s.parent.idx(id))
	return s.sparse[i/blockSize]&(1<<(i%blockSize)) != 0
}

// Elems calls f for each element of the set in ascending order.
func (s *PackageSet) Elems(f func(source.PackageID)) {
	blockIndexes := make([]int, 0, len(s.sparse))
	for k := range s.sparse {
		blockIndexes = append(blockIndexes, k)
	}
	sort.Ints(blockIndexes)
	for _, i := range blockIndexes {
		v := s.sparse[i]
		for b := 0; b < blockSize; b++ {
			if (v & (1 << b)) != 0 {
				f(s.parent.id(i*blockSize + b))
			}
		}
	}
}

// Len reports the length of the receiver set.
func (s *PackageSet) Len() int { // could be optimized
	l := 0
	s.Elems(func(source.PackageID) {
		l++
	})
	return l
}

// String returns a human-readable representation of the set: {A, B, ...}.
func (s *PackageSet) String() string {
	var ids []string
	s.Elems(func(id source.PackageID) {
		ids = append(ids, string(id))
	})
	return fmt.Sprintf("{%s}", strings.Join(ids, ", "))
}

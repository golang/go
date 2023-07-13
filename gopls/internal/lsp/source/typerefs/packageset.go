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
	m   map[source.PackageID]IndexID
}

// NewPackageIndex creates a new PackageIndex instance for use in building
// reference and package sets.
func NewPackageIndex() *PackageIndex {
	return &PackageIndex{
		m: make(map[source.PackageID]IndexID),
	}
}

// IndexID returns the packageIdx referencing id, creating one if id is not yet
// tracked by the receiver.
func (index *PackageIndex) IndexID(id source.PackageID) IndexID {
	index.mu.Lock()
	defer index.mu.Unlock()
	if i, ok := index.m[id]; ok {
		return i
	}
	i := IndexID(len(index.ids))
	index.m[id] = i
	index.ids = append(index.ids, id)
	return i
}

// PackageID returns the PackageID for idx.
//
// idx must have been created by this PackageIndex instance.
func (index *PackageIndex) PackageID(idx IndexID) source.PackageID {
	index.mu.Lock()
	defer index.mu.Unlock()
	return index.ids[idx]
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
func (index *PackageIndex) NewSet() *PackageSet {
	return &PackageSet{
		parent: index,
		sparse: make(map[int]blockType),
	}
}

// DeclaringPackage returns the ID of the symbol's declaring package.
// The package index must be the one used during decoding.
func (index *PackageIndex) DeclaringPackage(sym Symbol) source.PackageID {
	return index.PackageID(sym.Package)
}

// Add records a new element in the package set, for the provided package ID.
func (s *PackageSet) AddPackage(id source.PackageID) {
	s.Add(s.parent.IndexID(id))
}

// Add records a new element in the package set.
// It is the caller's responsibility to ensure that idx was created with the
// same PackageIndex as the PackageSet.
func (s *PackageSet) Add(idx IndexID) {
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
	i := int(s.parent.IndexID(id))
	return s.sparse[i/blockSize]&(1<<(i%blockSize)) != 0
}

// Elems calls f for each element of the set in ascending order.
func (s *PackageSet) Elems(f func(IndexID)) {
	blockIndexes := make([]int, 0, len(s.sparse))
	for k := range s.sparse {
		blockIndexes = append(blockIndexes, k)
	}
	sort.Ints(blockIndexes)
	for _, i := range blockIndexes {
		v := s.sparse[i]
		for b := 0; b < blockSize; b++ {
			if (v & (1 << b)) != 0 {
				f(IndexID(i*blockSize + b))
			}
		}
	}
}

// String returns a human-readable representation of the set: {A, B, ...}.
func (s *PackageSet) String() string {
	var ids []string
	s.Elems(func(id IndexID) {
		ids = append(ids, string(s.parent.PackageID(id)))
	})
	return fmt.Sprintf("{%s}", strings.Join(ids, ", "))
}

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"sort"

	"cmd/internal/obj"
)

// A Cache holds reusable compiler state.
// It is intended to be re-used for multiple Func compilations.
type Cache struct {
	// Storage for low-numbered values and blocks.
	values [2000]Value
	blocks [200]Block
	Locs   [2000]Location

	// Reusable stackAllocState.
	// See stackalloc.go's {new,put}StackAllocState.
	stackAllocState *StackAllocState

	scrPoset []*Poset // scratch poset to be reused

	// Reusable regalloc state.
	RegallocValues []ValState

	ValueToProgAfter []*obj.Prog
	DebugState       DebugState

	Liveness any // *gc.livenessFuncCache

	// Free "headers" for use by the allocators in allocators.go.
	// Used to put slices in sync.Pools without allocation.
	hdrValueSlice []*[]*Value
	hdrLimitSlice []*[]Limit
}

func (c *Cache) Reset() {
	nv := sort.Search(len(c.values), func(i int) bool { return c.values[i].ID == 0 })
	clear(c.values[:nv])
	nb := sort.Search(len(c.blocks), func(i int) bool { return c.blocks[i].ID == 0 })
	clear(c.blocks[:nb])
	nl := sort.Search(len(c.Locs), func(i int) bool { return c.Locs[i] == nil })
	clear(c.Locs[:nl])

	// regalloc sets the length of c.regallocValues to whatever it may use,
	// so clear according to length.
	clear(c.RegallocValues)
}

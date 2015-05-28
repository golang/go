// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

type ID int32

// idAlloc provides an allocator for unique integers.
type idAlloc struct {
	last ID
	free []ID
}

// get allocates an ID and returns it.
func (a *idAlloc) get() ID {
	if n := len(a.free); n > 0 {
		x := a.free[n-1]
		a.free = a.free[:n-1]
		return x
	}
	x := a.last
	x++
	if x == 1<<31-1 {
		panic("too many ids for this function")
	}
	a.last = x
	return x
}

// put deallocates an ID.
func (a *idAlloc) put(x ID) {
	a.free = append(a.free, x)
}

// num returns the maximum ID ever returned + 1.
func (a *idAlloc) num() int {
	return int(a.last + 1)
}

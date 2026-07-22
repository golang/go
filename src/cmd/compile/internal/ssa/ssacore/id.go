// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssacore

type ID int32

// IDAlloc provides an allocator for unique integers.
type IDAlloc struct {
	last ID
}

// Get allocates an ID and returns it. IDs are always > 0.
func (a *IDAlloc) Get() ID {
	x := a.last
	x++
	if x == 1<<31-1 {
		panic("too many ids for this function")
	}
	a.last = x
	return x
}

// Num returns the maximum ID ever returned + 1.
func (a *IDAlloc) Num() int {
	return int(a.last + 1)
}

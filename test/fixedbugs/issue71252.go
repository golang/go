// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// A multi-value assignment from a call to an instantiated generic
// function must not leave the caller's variable with the callee's shape
// type. Slicing a *[N]byte result used to fail with "bad ptr to array in
// slice go.shape.*uint8".

package main

import "fmt"

func peek[V any](v V) (V, bool) { return v, true }

type Cache[V any] struct{ v V }

func (c *Cache[V]) Peek() (V, bool) { return c.v, true }

type Rec struct{ Data []byte }

func main() {
	// A generic function returning (V, bool).
	blk, _ := peek(new([4096]byte))
	if got := len(blk[:]); got != 4096 {
		panic(fmt.Sprintf("len(blk[:]) = %d, want 4096", got))
	}

	// A method of a generic type.
	c := &Cache[*[2048]byte]{v: new([2048]byte)}
	blk2, _ := c.Peek()
	if got := len(blk2[:]); got != 2048 {
		panic(fmt.Sprintf("len(blk2[:]) = %d, want 2048", got))
	}

	// The slice used inside a composite literal.
	blk3, _ := peek(new([1024]byte))
	if got := len(Rec{Data: blk3[:]}.Data); got != 1024 {
		panic(fmt.Sprintf("len(Rec{...}.Data) = %d, want 1024", got))
	}
}

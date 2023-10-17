// run

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"runtime"
)

// linked list up the stack, to test lots of stack objects.

type T struct {
	// points to a heap object. Test will make sure it isn't freed.
	data *int64
	// next pointer for a linked list of stack objects
	next *T
	// duplicate of next, to stress test the pointer buffers
	// used during stack tracing.
	next2 *T
}

func main() {
	makelist(nil, 10000)
}

func makelist(x *T, n int64) {
	if n%2 != 0 {
		panic("must be multiple of 2")
	}
	if n == 0 {
		runtime.GC()
		i := int64(1)
		for ; x != nil; x, i = x.next, i+1 {
			// Make sure x.data hasn't been collected.
			if got := *x.data; got != i {
				panic(fmt.Sprintf("bad data want %d, got %d", i, got))
			}
		}
		return
	}
	// Put 2 objects in each frame, to test intra-frame pointers.
	// Use both orderings to ensure the linked list isn't always in address order.
	var a, b T
	if n%3 == 0 {
		a.data = newInt(n)
		a.next = x
		a.next2 = x
		b.data = newInt(n - 1)
		b.next = &a
		b.next2 = &a
		x = &b
	} else {
		b.data = newInt(n)
		b.next = x
		b.next2 = x
		a.data = newInt(n - 1)
		a.next = &b
		a.next2 = &b
		x = &a
	}

	makelist(x, n-2)
}

// big enough and pointer-y enough to not be tinyalloc'd
type NotTiny struct {
	n int64
	p *byte
}

// newInt allocates n on the heap and returns a pointer to it.
func newInt(n int64) *int64 {
	h := &NotTiny{n: n}
	p := &h.n
	escape = p
	return p
}

var escape *int64

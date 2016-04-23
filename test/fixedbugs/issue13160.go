// run

// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"runtime"
)

const N = 100000

func main() {
	// Allocate more Ps than processors.  This raises
	// the chance that we get interrupted by the OS
	// in exactly the right (wrong!) place.
	p := runtime.NumCPU()
	runtime.GOMAXPROCS(2 * p)

	// Allocate some pointers.
	ptrs := make([]*int, p)
	for i := 0; i < p; i++ {
		ptrs[i] = new(int)
	}

	// Arena where we read and write pointers like crazy.
	collider := make([]*int, p)

	done := make(chan struct{}, 2*p)

	// Start writers.  They alternately write a pointer
	// and nil to a slot in the collider.
	for i := 0; i < p; i++ {
		i := i
		go func() {
			for j := 0; j < N; j++ {
				// Write a pointer using memmove.
				copy(collider[i:i+1], ptrs[i:i+1])
				// Write nil using memclr.
				// (This is a magic loop that gets lowered to memclr.)
				r := collider[i : i+1]
				for k := range r {
					r[k] = nil
				}
			}
			done <- struct{}{}
		}()
	}
	// Start readers.  They read pointers from slots
	// and make sure they are valid.
	for i := 0; i < p; i++ {
		i := i
		go func() {
			for j := 0; j < N; j++ {
				var ptr [1]*int
				copy(ptr[:], collider[i:i+1])
				if ptr[0] != nil && ptr[0] != ptrs[i] {
					panic(fmt.Sprintf("bad pointer read %p!", ptr[0]))
				}
			}
			done <- struct{}{}
		}()
	}
	for i := 0; i < 2*p; i++ {
		<-done
	}
}

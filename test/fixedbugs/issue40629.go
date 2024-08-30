// run

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "fmt"

const N = 40

func main() {
	var x [N]int // stack-allocated memory
	for i := range x {
		x[i] = 0x999
	}

	// This defer checks to see if x is uncorrupted.
	defer func(p *[N]int) {
		recover()
		for i := range p {
			if p[i] != 0x999 {
				for j := range p {
					fmt.Printf("p[%d]=0x%x\n", j, p[j])
				}
				panic("corrupted stack variable")
			}
		}
	}(&x)

	// This defer starts a new goroutine, which will (hopefully)
	// overwrite x on the garbage stack.
	defer func() {
		c := make(chan bool)
		go func() {
			useStack(1000)
			c <- true
		}()
		<-c

	}()

	// This defer causes a stack copy.
	// The old stack is now garbage.
	defer func() {
		useStack(1000)
	}()

	// Trigger a segfault.
	*g = 0

	// Make the return statement unreachable.
	// That makes the stack map at the deferreturn call empty.
	// In particular, the argument to the first defer is not
	// marked as a pointer, so it doesn't get adjusted
	// during the stack copy.
	for {
	}
}

var g *int64

func useStack(n int) {
	if n == 0 {
		return
	}
	useStack(n - 1)
}

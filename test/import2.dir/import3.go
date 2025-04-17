// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that all the types from import2.go made it
// intact and with the same meaning, by assigning to or using them.

package main

import "./import2"

func f3(func() func() int)

func main() {
	p.F3(p.F1)
	p.F3(p.F2())
	f3(p.F1)
	f3(p.F2())

	p.C1 = (chan<- (chan int))(nil)
	p.C2 = (chan (<-chan int))(nil)
	p.C3 = (<-chan (chan int))(nil)
	p.C4 = (chan (chan<- int))(nil)

	p.C5 = (<-chan (<-chan int))(nil)
	p.C6 = (chan<- (<-chan int))(nil)
	p.C7 = (chan<- (chan<- int))(nil)

	p.C8 = (<-chan (<-chan (chan int)))(nil)
	p.C9 = (<-chan (chan<- (chan int)))(nil)
	p.C10 = (chan<- (<-chan (chan int)))(nil)
	p.C11 = (chan<- (chan<- (chan int)))(nil)
	p.C12 = (chan (chan<- (<-chan int)))(nil)
	p.C13 = (chan (chan<- (chan<- int)))(nil)

	p.R1 = (chan <- chan int)(nil)
	p.R3 = (<- chan chan int)(nil)
	p.R4 = (chan chan <- int)(nil)

	p.R5 = (<- chan <- chan int)(nil)
	p.R6 = (chan <- <- chan int)(nil)
	p.R7 = (chan <- chan <- int)(nil)

	p.R8 = (<- chan <- chan chan int)(nil)
	p.R9 = (<- chan chan <- chan int)(nil)
	p.R10 = (chan <- <- chan chan int)(nil)
	p.R11 = (chan <- chan <- chan int)(nil)
	p.R12 = (chan chan <- <- chan int)(nil)
	p.R13 = (chan chan <- chan <- int)(nil)

}


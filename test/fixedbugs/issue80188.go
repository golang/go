// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "runtime"

type T struct {
	a, b, c, d int
}

func (t *T) sum() int {
	return t.a + t.b + t.c + t.d
}

//go:noinline
func newT(x int) *T {
	t := new(T)
	t.a = x
	t.b = x
	t.c = x
	t.d = x
	return t
}

//go:noinline
func intptr() *int {
	return &g
}

var g int

//go:noinline
func f(x int) []*T {
	const N = 40
	var q [N]*T
	for i := range N {
		q[i] = newT(x)
	}
	var a [2]*int
	for i := range a {
		a[i] = intptr()
	}

	// reserve 2 registers we will later drop.
	b := a[0]
	c := a[1]

	// Load pointers into registers (or spill slots)
	p0 := q[0]
	p1 := q[1]
	p2 := q[2]
	p3 := q[3]
	p4 := q[4]
	p5 := q[5]
	p6 := q[6]
	p7 := q[7]
	p8 := q[8]
	p9 := q[9]
	p10 := q[10]
	p11 := q[11]
	p12 := q[12]
	p13 := q[13]
	p14 := q[14]
	p15 := q[15]
	p16 := q[16]
	p17 := q[17]
	p18 := q[18]
	p19 := q[19]
	p20 := q[20]
	p21 := q[21]
	p22 := q[22]
	p23 := q[23]
	p24 := q[24]
	p25 := q[25]
	p26 := q[26]
	p27 := q[27]
	p28 := q[28]
	p29 := q[29]
	p30 := q[30]
	p31 := q[31]
	p32 := q[32]
	p33 := q[33]
	p34 := q[34]
	p35 := q[35]
	p36 := q[36]
	p37 := q[37]
	p38 := q[38]
	p39 := q[39]

	// q is dead at this point. The only live reference to
	// the objects allocated by newT are in the pXX variables.
	// But if async preempted, we will scan the frame conservatively,
	// which will find dead entries in q. Remove them.
	q[0] = nil
	q[1] = nil
	q[2] = nil
	q[3] = nil
	q[4] = nil
	q[5] = nil
	q[6] = nil
	q[7] = nil
	q[8] = nil
	q[9] = nil
	q[10] = nil
	q[11] = nil
	q[12] = nil
	q[13] = nil
	q[14] = nil
	q[15] = nil
	q[16] = nil
	q[17] = nil
	q[18] = nil
	q[19] = nil
	q[20] = nil
	q[21] = nil
	q[22] = nil
	q[23] = nil
	q[24] = nil
	q[25] = nil
	q[26] = nil
	q[27] = nil
	q[28] = nil
	q[29] = nil
	q[30] = nil
	q[31] = nil
	q[32] = nil
	q[33] = nil
	q[34] = nil
	q[35] = nil
	q[36] = nil
	q[37] = nil
	q[38] = nil
	q[39] = nil

	// Some pXX is held in R30.
	// If async preemption happens here, the pointer in R30
	// will not get scanned and might cause the pointed-to
	// object to be collected prematurely. It does not live
	// anywhere else in this frame, and we will not scan
	// this frame again later.
	//
	// Note that the write barrier has to be off for some of
	// the newT calls for objects not to be allocated black.
	// But it must be on here to get an async preempt.
	// It must be off again by the final copy to not get put
	// in a write barrier buffer somewhere.

	// delay here to encourage async preemption
	*b = 0
	*c = 0
	// The 2 registers holding b,c are now free. They
	// can be used to implement this loop without spilling
	// R30 to get a free register.
	for range 10000 {
	}

	// Store registers back to stack memory (with no write barriers)
	var z [N]*T
	z[0] = p0
	z[1] = p1
	z[2] = p2
	z[3] = p3
	z[4] = p4
	z[5] = p5
	z[6] = p6
	z[7] = p7
	z[8] = p8
	z[9] = p9
	z[10] = p10
	z[11] = p11
	z[12] = p12
	z[13] = p13
	z[14] = p14
	z[15] = p15
	z[16] = p16
	z[17] = p17
	z[18] = p18
	z[19] = p19
	z[20] = p20
	z[21] = p21
	z[22] = p22
	z[23] = p23
	z[24] = p24
	z[25] = p25
	z[26] = p26
	z[27] = p27
	z[28] = p28
	z[29] = p29
	z[30] = p30
	z[31] = p31
	z[32] = p32
	z[33] = p33
	z[34] = p34
	z[35] = p35
	z[36] = p36
	z[37] = p37
	z[38] = p38
	z[39] = p39

	// Delay again, hoping GC finishes before we publish
	// the unscanned pointer via the write barrier.
	for range 10000 {
	}

	// Copy results to heap to return.
	r := new([N]*T)
	*r = z
	return r[:]
}

func main() {
	runtime.GOMAXPROCS(2)
	c := make(chan bool)
	for range 4 {
		go goroutine(c)
	}
	for range 4 {
		<-c
	}
}
func goroutine(c chan bool) {
	var all [][]*T
	for x := range 10000 {
		all = append(all, f(x))
	}
	for x, a := range all {
		for _, t := range a {
			if t.a != x || t.b != x || t.c != x || t.d != x {
				panic("bad")
			}
		}
	}
	c <- true
}

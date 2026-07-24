// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Proc unit tests. In runtime package so can use runtime guts.

package runtime

func RunStealOrderTest() {
	var ord randomOrder
	for procs := 1; procs <= 64; procs++ {
		ord.reset(uint32(procs))
		if procs >= 3 && len(ord.coprimes) < 2 {
			panic("too few coprimes")
		}
		for co := 0; co < len(ord.coprimes); co++ {
			iter := ord.start(uint32(co))
			checked := make([]bool, procs)
			for p := 0; p < procs; p++ {
				x := iter.position()
				if checked[x] {
					println("procs:", procs, "inc:", iter.inc)
					panic("duplicate during enumeration")
				}
				checked[x] = true
				iter.next()
			}
			if !iter.done() {
				panic("not done")
			}
		}
	}
	// Make sure that different arguments to ord.start don't generate the
	// same pos+inc twice.
	for procs := 2; procs <= 64; procs++ {
		ord.reset(uint32(procs))
		checked := make([]bool, procs*procs)
		// We want at least procs*len(ord.coprimes) different pos+inc values
		// before we start repeating.
		for i := 0; i < procs*len(ord.coprimes); i++ {
			iter := ord.start(uint32(i))
			j := iter.pos*uint32(procs) + iter.inc
			if checked[j] {
				println("procs:", procs, "pos:", iter.pos, "inc:", iter.inc)
				panic("duplicate pos+inc during enumeration")
			}
			checked[j] = true
		}
	}
}

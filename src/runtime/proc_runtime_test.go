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
			enum := ord.start(uint32(co))
			checked := make([]bool, procs)
			for p := 0; p < procs; p++ {
				x := enum.position()
				if checked[x] {
					println("procs:", procs, "inc:", enum.inc)
					panic("duplicate during enumeration")
				}
				checked[x] = true
				enum.next()
			}
			if !enum.done() {
				panic("not done")
			}
		}
	}
}

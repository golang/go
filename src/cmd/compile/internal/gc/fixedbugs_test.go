// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import "testing"

type T struct {
	x [2]int64 // field that will be clobbered. Also makes type not SSAable.
	p *byte    // has a pointer
}

//go:noinline
func makeT() T {
	return T{}
}

var g T

var sink []byte

func TestIssue15854(t *testing.T) {
	for i := 0; i < 10000; i++ {
		if g.x[0] != 0 {
			t.Fatalf("g.x[0] clobbered with %x\n", g.x[0])
		}
		// The bug was in the following assignment. The return
		// value of makeT() is not copied out of the args area of
		// stack frame in a timely fashion. So when write barriers
		// are enabled, the marshaling of the args for the write
		// barrier call clobbers the result of makeT() before it is
		// read by the write barrier code.
		g = makeT()
		sink = make([]byte, 1000) // force write barriers to eventually happen
	}
}
func TestIssue15854b(t *testing.T) {
	const N = 10000
	a := make([]T, N)
	for i := 0; i < N; i++ {
		a = append(a, makeT())
		sink = make([]byte, 1000) // force write barriers to eventually happen
	}
	for i, v := range a {
		if v.x[0] != 0 {
			t.Fatalf("a[%d].x[0] clobbered with %x\n", i, v.x[0])
		}
	}
}

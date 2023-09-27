// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// WARNING: Please avoid updating this file. If this file needs to be updated,
// then a new devirt.pprof file should be generated:
//
//	$ cd $GOROOT/src/cmd/compile/internal/test/testdata/pgo/devirtualize/
//	$ go mod init example.com/pgo/devirtualize
//	$ go test -bench=. -cpuprofile ./devirt.pprof

package devirt

import "example.com/pgo/devirtualize/mult"

var sink int

type Adder interface {
	Add(a, b int) int
}

type Add struct{}

func (Add) Add(a, b int) int {
	for i := 0; i < 1000; i++ {
		sink++
	}
	return a + b
}

type Sub struct{}

func (Sub) Add(a, b int) int {
	for i := 0; i < 1000; i++ {
		sink++
	}
	return a - b
}

// Exercise calls mostly a1 and m1.
//
//go:noinline
func Exercise(iter int, a1, a2 Adder, m1, m2 mult.Multiplier) {
	for i := 0; i < iter; i++ {
		a := a1
		m := m1
		if i%10 == 0 {
			a = a2
			m = m2
		}

		// N.B. Profiles only distinguish calls on a per-line level,
		// making the two calls ambiguous. However because the
		// interfaces and implementations are mutually exclusive,
		// devirtualization can still select the correct callee for
		// each.
		//
		// If they were not mutually exclusive (for example, two Add
		// calls), then we could not definitively select the correct
		// callee.
		sink += m.Multiply(42, a.Add(1, 2))
	}
}

func init() {
	// TODO: until https://golang.org/cl/497175 or similar lands,
	// we need to create an explicit reference to callees
	// in another package for devirtualization to work.
	m := mult.Mult{}
	m.Multiply(42, 0)
	n := mult.NegMult{}
	n.Multiply(42, 0)
}

// run

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"math"
)

type S struct {
	u int64
	n int32
}

func F1(f float64) *S {
	s := f
	pf := math.Copysign(f, 1)
	u := math.Floor(pf)
	return &S{
		u: int64(math.Copysign(u, s)),
		n: int32(math.Copysign((pf-u)*1e9, s)),
	}
}

func F2(f float64) *S {
	s := f
	f = math.Copysign(f, 1)
	u := math.Floor(f)
	return &S{
		u: int64(math.Copysign(u, s)),
		n: int32(math.Copysign((f-u)*1e9, s)),
	}
}

func main() {
	s1 := F1(-1)
	s2 := F2(-1)
	if *s1 != *s2 {
		println("F1:", s1.u, s1.n)
		println("F2:", s2.u, s2.n)
		panic("different")
	}
}

// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// trivial finalizer test

package main

import "runtime"

const N = 250

type A struct {
	b *B
	n int
}

type B struct {
	n int
}

var i int
var nfinal int
var final [N]int

func finalA(a *A) {
	if final[a.n] != 0 {
		panicln("finalA", a.n, final[a.n])
	}
	final[a.n] = 1
}

func finalB(b *B) {
	if final[b.n] != 1 {
		panicln("finalB", b.n, final[b.n])
	}
	final[b.n] = 2
	nfinal++
}

func main() {
	runtime.GOMAXPROCS(4)
	for i = 0; i < N; i++ {
		b := &B{i}
		a := &A{b, i}
		runtime.SetFinalizer(b, finalB)
		runtime.SetFinalizer(a, finalA)
	}
	for i := 0; i < N; i++ {
		runtime.GC()
		runtime.Gosched()
	}
	if nfinal < N*9/10 {
		panic("not enough finalizing:", nfinal, "/", N)
	}
}

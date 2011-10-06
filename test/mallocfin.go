// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// trivial finalizer test

package main

import (
	"runtime"
	"time"
)

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

// the unused return is to test finalizers with return values
func finalA(a *A) (unused [N]int) {
	if final[a.n] != 0 {
		println("finalA", a.n, final[a.n])
		panic("fail")
	}
	final[a.n] = 1
	return
}

func finalB(b *B) {
	if final[b.n] != 1 {
		println("finalB", b.n, final[b.n])
		panic("fail")
	}
	final[b.n] = 2
	nfinal++
}

func nofinalB(b *B) {
	panic("nofinalB run")
}

func main() {
	runtime.GOMAXPROCS(4)
	for i = 0; i < N; i++ {
		b := &B{i}
		a := &A{b, i}
		c := new(B)
		runtime.SetFinalizer(c, nofinalB)
		runtime.SetFinalizer(b, finalB)
		runtime.SetFinalizer(a, finalA)
		runtime.SetFinalizer(c, nil)
	}
	for i := 0; i < N; i++ {
		runtime.GC()
		runtime.Gosched()
		time.Sleep(1e6)
		if nfinal >= N*8/10 {
			break
		}
	}
	if nfinal < N*8/10 {
		println("not enough finalizing:", nfinal, "/", N)
		panic("fail")
	}
}

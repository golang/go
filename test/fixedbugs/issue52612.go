// run

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"sync/atomic"
	"unsafe"
)

var one interface{} = 1

type eface struct {
	typ  unsafe.Pointer
	data unsafe.Pointer
}

func f(c chan struct{}) {
	var x atomic.Value

	go func() {
		x.Swap(one) // writing using the old marker
	}()
	for i := 0; i < 100000; i++ {
		v := x.Load() // reading using the new marker

		p := (*eface)(unsafe.Pointer(&v)).typ
		if uintptr(p) == ^uintptr(0) {
			// We read the old marker, which the new reader
			// doesn't know is a case where it should retry
			// instead of returning it.
			panic("bad typ field")
		}
	}
	c <- struct{}{}
}

func main() {
	c := make(chan struct{}, 10)
	for i := 0; i < 10; i++ {
		go f(c)
	}
	for i := 0; i < 10; i++ {
		<-c
	}
}

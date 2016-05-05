// skip

// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package life

// #include "life.h"
import "C"

import "unsafe"

func Run(gen, x, y int, a []int32) {
	n := make([]int32, x*y)
	for i := 0; i < gen; i++ {
		C.Step(C.int(x), C.int(y), (*C.int)(unsafe.Pointer(&a[0])), (*C.int)(unsafe.Pointer(&n[0])))
		copy(a, n)
	}
}

// Keep the channels visible from Go.
var chans [4]chan bool

//export GoStart
// Double return value is just for testing.
func GoStart(i, xdim, ydim, xstart, xend, ystart, yend C.int, a *C.int, n *C.int) (int, int) {
	c := make(chan bool, int(C.MYCONST))
	go func() {
		C.DoStep(xdim, ydim, xstart, xend, ystart, yend, a, n)
		c <- true
	}()
	chans[i] = c
	return int(i), int(i + 100)
}

//export GoWait
func GoWait(i C.int) {
	<-chans[i]
	chans[i] = nil
}

// run

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"strconv"
	"sync/atomic"
	"unsafe"
)

type t struct {
	i1 atomic.Int32
	i2 atomic.Int64
}

var v t

func main() {
	if o := unsafe.Offsetof(v.i2); o != 8 {
		panic("unexpected offset, want: 8, got: " + strconv.Itoa(int(o)))
	}
}

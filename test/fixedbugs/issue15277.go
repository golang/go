// run

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
// +build amd64

package main

import "runtime"

type big [10 << 20]byte

func f(x *big, start int64) {
	if delta := inuse() - start; delta < 9<<20 {
		println("after alloc: expected delta at least 9MB, got: ", delta)
	}
	runtime.KeepAlive(x)
	x = nil
	if delta := inuse() - start; delta > 1<<20 {
		println("after drop: expected delta below 1MB, got: ", delta)
	}
	x = new(big)
	if delta := inuse() - start; delta < 9<<20 {
		println("second alloc: expected delta at least 9MB, got: ", delta)
	}
	runtime.KeepAlive(x)
}

func main() {
	x := inuse()
	f(new(big), x)
}

func inuse() int64 {
	runtime.GC()
	var st runtime.MemStats
	runtime.ReadMemStats(&st)
	return int64(st.Alloc)
}

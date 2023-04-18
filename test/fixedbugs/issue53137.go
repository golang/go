// run

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"unsafe"
)

type Embedded struct {
	B int
}

type S[K any] struct {
	A K
	Embedded
}

func showOffsets[K any](d *S[K]) {
	o1 := unsafe.Offsetof(d.B)
	o2 := unsafe.Offsetof(d.Embedded)
	if o1 != o2 {
		panic("offset mismatch")
	}
}

func main() {
	showOffsets(new(S[int]))
}

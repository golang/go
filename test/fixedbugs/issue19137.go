// compile

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 19137: folding address into load/store causes
// odd offset on ARM64.

package p

type T struct {
	p *int
	a [2]byte
	b [6]byte // not 4-byte aligned
}

func f(b [6]byte) T {
	var x [1000]int // a large stack frame
	_ = x
	return T{b: b}
}

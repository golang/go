// compile

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type S struct {
	a [4]struct{}
	f chan int
}

func f(p *S) {
	var s S

	// Memory write that requires a write barrier should work
	// with structs having zero-sized arrays of non-zero elements.
	*p = s
}

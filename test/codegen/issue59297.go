// asmcheck

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

func f(x, y int, p *int) {
	// amd64:`MOVQ\sAX, BX`
	h(8, x)
	*p = y
}

//go:noinline
func h(a, b int) {
}

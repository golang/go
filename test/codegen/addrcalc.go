// asmcheck

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

// Make sure we use ADDQ instead of LEAQ when we can.

func f(p *[4][2]int, x int) *int {
	// amd64:"ADDQ",-"LEAQ"
	return &p[x][0]
}

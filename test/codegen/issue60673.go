// asmcheck

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

//go:noinline
func f(x int32) {
}

func g(p *int32) {
	// argument marshaling code should live at line 17, not line 15.
	x := *p
	// 386: `MOVL\s[A-Z]+,\s\(SP\)`
	f(x)
}

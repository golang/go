// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "./lib"

func main() {
	r := &lib.Reader{B: 5}
	if lib.Direct(r) != 5 || lib.Direct((*lib.Reader)(nil)) != 9 {
		panic("direct or nil receiver")
	}
	if lib.Forward(r) != 5 || lib.ViaAlias(r) != 5 {
		panic("alias constraint or generic forwarding")
	}
	if lib.TheValue(r)() != 5 || lib.TheExpr(r)(r) != 5 {
		panic("method value or expression")
	}
	if lib.Union(r) != 5 || lib.Union(&lib.OtherReader{B: 5}) != 6 {
		panic("union method selection")
	}
	if lib.Promoted(&lib.Embedded{Reader: r}) != 5 {
		panic("promoted method")
	}
}

// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "./a"

func main() {
	for _, f := range []func() int{
		a.F1, a.F2, a.F3, a.F4,
		a.F5, a.F6, a.F7, a.F8, a.F9} {
		if f() > 1 {
			panic("f() > 1")
		}
	}
}

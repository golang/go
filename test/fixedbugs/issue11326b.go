// run

// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	/* TODO(rsc): Should work but does not. See golang.org/issue/11326.
	{
		const n = 1e2147483647
		const d = 1e2147483646
		x := n / d
		if x != 10.0 {
			println("incorrect value:", x)
		}
	}
	{
		const n = 1e214748364
		const d = 1e214748363
		x := n / d
		if x != 10.0 {
			println("incorrect value:", x)
		}
	}
	*/
	{
		const n = 1e21474836
		const d = 1e21474835
		x := n / d
		if x != 10.0 {
			println("incorrect value:", x)
		}
	}
	{
		const n = 1e2147483
		const d = 1e2147482
		x := n / d
		if x != 10.0 {
			println("incorrect value:", x)
		}
	}
}

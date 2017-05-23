// run

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 11987. The ppc64 SRADCC instruction was misassembled in a way
// lost bit 5 of the immediate so v>>32 was assembled as v>>0.  SRADCC
// is only ever inserted by peep so it's hard to be sure when it will
// be used. This formulation worked when the bug was fixed.

package main

import "fmt"

var v int64 = 0x80000000

func main() {
	s := fmt.Sprintf("%v", v>>32 == 0)
	if s != "true" {
		fmt.Printf("BUG: v>>32 == 0 evaluated as %q\n", s)
	}
}

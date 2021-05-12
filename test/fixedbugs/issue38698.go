// compile

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This test case caused a panic in the compiler's DWARF gen code.

package p

func ff( /*line :10*/ x string) bool {
	{
		var _ /*line :10*/, x int
		_ = x
	}
	return x == ""
}


func h(a string) bool {
	return ff(a)
}


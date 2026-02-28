// compile

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The gccgo compiler crashed while compiling this code.
// https://gcc.gnu.org/PR78763.

package p

import "unsafe"

func F() int {
	if unsafe.Sizeof(0) == 8 {
		return 8
	}
	return 0
}

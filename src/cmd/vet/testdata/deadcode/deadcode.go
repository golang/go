// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains tests for the dead code checker.

package deadcode

func _() int {
	print(1)
	return 2
	println() // ERROR "unreachable code"
	return 3
}

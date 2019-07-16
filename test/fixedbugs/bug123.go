// errorcheck

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main
const ( F = 1 )
func fn(i int) int {
	if i == F() {		// ERROR "func"
		return 0
	}
	return 1
}

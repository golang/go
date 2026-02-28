// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package exp

func Exported(x int) int {
	return inlined(x)
}

func inlined(x int) int {
	y := 0
	switch {
	case x > 0:
		y += 5
		return 0 + y
	case x < 1:
		y += 6
		fallthrough
	default:
		y += 7
		return 2 + y
	}
}

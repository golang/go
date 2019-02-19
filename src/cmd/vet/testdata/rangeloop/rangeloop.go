// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains tests for the rangeloop checker.

package rangeloop

func RangeLoopTests() {
	var s []int
	for i, v := range s {
		go func() {
			println(i) // ERROR "loop variable i captured by func literal"
			println(v) // ERROR "loop variable v captured by func literal"
		}()
	}
}

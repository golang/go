// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package race_test

// golang.org/issue/13264
// The test is that this compiles at all.

func issue13264() {
	for ; ; []map[int]int{}[0][0] = 0 {
	}
}

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// We want the initializers of these packages to occur in source code
// order. See issue 31636. This is the behavior up to and including
// 1.13. For 1.14, we will move to a variant of lexicographic ordering
// which will require a change to the test output of this test.
import (
	_ "./c"

	_ "./b"

	_ "./a"
)

func main() {
}

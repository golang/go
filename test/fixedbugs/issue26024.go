// compile
// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package r

// f compiles into code where no stores remain in the two successors
// of a write barrier block; i.e., they are empty. Pre-fix, this
// results in an unexpected input to markUnsafePoints, that expects to
// see a pair of non-empty plain blocks.
func f() {
	var i int
	var s string
	for len(s) < len(s) {
		i++
		s = "a"
	}
	var b bool
	var sa []string
	for true {
		sa = []string{""}
		for b || i == 0 {
		}
		b = !b
		_ = sa
	}
}

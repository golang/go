// errorcheck

// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 873, 2162

package foo

func f(x interface{}) {
	switch t := x.(type) {  // ERROR "declared and not used"
	case int:
	}
}

func g(x interface{}) {
	switch t := x.(type) {
	case int:
	case float32:
		println(t)
	}
}

func h(x interface{}) {
	switch t := x.(type) {
	case int:
	case float32:
	default:
		println(t)
	}
}

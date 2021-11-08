// errorcheck -d=panic

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Verify that we get "use of .(type) outside type switch"
// before any other (misleading) errors. Test case from issue.

package p

func f(i interface{}) {
	if x, ok := i.(type); ok { // ERROR "assignment mismatch|outside type switch"
		_ = x
	}
}

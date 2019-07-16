// errorcheck

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Check for cycles in a function value.

package funcvalue

func fx() int {
	_ = x
	return 0
}

var x = fx // ERROR "initialization loop|depends upon itself"

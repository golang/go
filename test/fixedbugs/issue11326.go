// errorcheck

// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests for golang.org/issue/11326.

package main

func main() {
	var _ = 1e2147483647 // ERROR "constant too large"
	var _ = 1e646456993  // ERROR "constant too large"
	var _ = 1e646456992  // ERROR "1.00000e\+646456992 overflows float64"
	var _ = 1e64645699   // ERROR "1.00000e\+64645699 overflows float64"
	var _ = 1e6464569    // ERROR "1.00000e\+6464569 overflows float64"
	var _ = 1e646456     // ERROR "1.00000e\+646456 overflows float64"
	var _ = 1e64645      // ERROR "1.00000e\+64645 overflows float64"
	var _ = 1e6464       // ERROR "1.00000e\+6464 overflows float64"
	var _ = 1e646        // ERROR "1.00000e\+646 overflows float64"
	var _ = 1e309        // ERROR "1.00000e\+309 overflows float64"
	var _ = 1e308
}

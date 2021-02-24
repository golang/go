// errorcheck

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Verify that a label named like a package is recognized
// as a label rather than a package and that the package
// remains unused.

package main

import "math" // ERROR "imported and not used|imported but not used"

func main() {
math:
	for {
		break math
	}
}

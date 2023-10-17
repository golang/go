// errorcheck -0 -m -l

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test escape analysis for hash/maphash.

package escape

import (
	"hash/maphash"
)

func f() {
	var x maphash.Hash // should be stack allocatable
	x.WriteString("foo")
	x.Sum64()
}

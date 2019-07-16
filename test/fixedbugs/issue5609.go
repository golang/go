// errorcheck

// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// issue 5609: overflow when calculating array size

package pkg

const Large uint64 = 18446744073709551615

var foo [Large]uint64 // ERROR "array bound is too large|array bound overflows"

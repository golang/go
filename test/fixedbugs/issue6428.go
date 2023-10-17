// errorcheck

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

import . "testing" // ERROR "imported and not used"

type S struct {
	T int
}

var _ = S{T: 0}

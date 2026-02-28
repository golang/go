// build

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

import "math"

func f() {
	_ = min(0.1, 0.2, math.Sqrt(1))
}

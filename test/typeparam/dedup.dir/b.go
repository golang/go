// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package b

import "./a"

func B() {
	var x int64
	println(a.F(&x, &x))
	var y int32
	println(a.F(&y, &y))
}

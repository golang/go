// compile

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

var x = f(-1)
var y = f(64)

func f(x int) int {
	return 1 << x
}

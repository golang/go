// compile

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 16804: internal error for math.Sqrt as statement
//              rather than expression

package main

import "math"

func sqrt() {
	math.Sqrt(2.0)
}

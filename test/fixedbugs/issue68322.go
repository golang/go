// run

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "math"

var doNotFold = 18446744073709549568.0

func main() {
	if math.Trunc(doNotFold) != doNotFold {
		panic("big (over 2**63-1) math.Trunc is incorrect")
	}
}

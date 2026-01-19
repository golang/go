// compile

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"math"
)

func main() {
	test(2)
}

func test(i int) {
	if i <= 0 {
		return
	}

	_ = math.Pow10(i + 2)
}

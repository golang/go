// compile

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "math"

func main() {
	f := func(p bool) {
		if p {
			println("hi")
		}
	}
	go f(true || math.Sqrt(2) > 1)
}

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "./a" // import must succeed

func main() {
	if a.F()(a.T{}) != "m" {
		panic(0)
	}
	if a.Fp()(nil) != "mp" {
		panic(1)
	}
}

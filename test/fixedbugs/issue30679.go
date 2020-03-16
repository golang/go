// compile

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	var f float64
	var p, q *float64

	p = &f
	if *q > 0 {
		p = q
	}
	_ = *p
}

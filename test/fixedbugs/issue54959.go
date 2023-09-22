// run

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var p *int

func main() {
	var i int
	p = &i // escape i to keep the compiler from making the closure trivial

	func() { i++ }()
}

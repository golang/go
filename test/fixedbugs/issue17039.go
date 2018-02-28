// run

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type S []S

func main() {
	var s S
	s = append(s, s) // append a nil value to s
	if s[0] != nil {
		println("BUG: s[0] != nil")
	}
}

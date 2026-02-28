// errorcheck

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func f1() {
exit:
	print("hi\n")
	goto exit
}

func f2() {
	const c = 1234
}

func f3() {
	i := c // ERROR "undef"
	_ = i
}

func main() {
	f3()
}

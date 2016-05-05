// compile

// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 4399: 8g would print "gins LEAQ nil *A".

package main

type A struct{ a int }

func main() {
	println(((*A)(nil)).a)
}

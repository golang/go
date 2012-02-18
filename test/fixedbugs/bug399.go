// compile

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 2674

package main
const dow = "\000\003"

func main() {
	println(int(dow[1]))
}


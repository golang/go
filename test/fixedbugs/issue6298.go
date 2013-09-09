// compile

// golang.org/issue/6298.
// Used to cause "internal error: typename ideal bool"

// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	var x interface{} = "abc"[0] == 'a'
	_ = x
}

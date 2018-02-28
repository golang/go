// compile

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// issue 7366: generates a temporary with ideal type
// during comparison of small structs.

package main

type T struct {
	data [10]byte
}

func main() {
	var a T
	var b T
	if a == b {
	}
}

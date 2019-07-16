// run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	var a [1000] int64;  // this alone works
	var b [10000] int64;  // this causes a runtime crash
	_, _ = a, b;
}

/*
uetli:~/Source/go1/test/bugs gri$ 6g bug092.go && 6l bug092.6 && 6.out
Illegal instruction

gri: array size matters, possibly related to stack overflow check?
*/

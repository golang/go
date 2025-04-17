// errorcheck

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func f() int {
	if false {
		return 0;
	}
	// we should not be able to return successfully w/o a return statement
} // ERROR "return"

func main() {
	print(f(), "\n");
}

/*
uetli:~/Source/go1/usr/gri/gosrc gri$ 6g bug.go && 6l bug.6 && 6.out
4882
*/

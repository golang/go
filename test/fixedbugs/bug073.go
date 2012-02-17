// errorcheck

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	var s int = 0;
	var x int = 0;
	x = x << s;  // ERROR "illegal|inval|shift"
	x = x >> s;  // ERROR "illegal|inval|shift"
}

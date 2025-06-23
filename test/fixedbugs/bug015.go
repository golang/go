// errorcheck

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	var i33 int64;
	if i33 == (1<<64) -1 {  // ERROR "overflow"
	}
}

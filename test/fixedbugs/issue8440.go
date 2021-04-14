// errorcheck

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	n.foo = 6 // ERROR "undefined: n in n.foo|undefined name .*n"
}

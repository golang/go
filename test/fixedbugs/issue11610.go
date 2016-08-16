// errorcheck

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test an internal compiler error on ? symbol in declaration
// following an empty import.

package a
import""  // ERROR "import path is empty"
var?      // ERROR "illegal character U\+003F '\?'"

var x int // ERROR "unexpected var"

func main() {
}

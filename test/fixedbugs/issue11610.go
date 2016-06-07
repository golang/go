// errorcheck -newparser=0

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test an internal compiler error on ? symbol in declaration
// following an empty import.

// TODO(mdempsky): Update for new parser. New parser recovers more
// gracefully and doesn't trigger the "cannot declare name" error.
// Also remove "errorcheck -newparser=0" case in go/types.TestStdFixed.

package a
import""  // ERROR "import path is empty"
var?      // ERROR "illegal character U\+003F '\?'"

var x int // ERROR "unexpected var" "cannot declare name"

func main() {
}

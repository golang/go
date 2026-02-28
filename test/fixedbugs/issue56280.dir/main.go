// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "test/a"

func main() { // ERROR "can inline main"
	a.F() // ERROR "inlining call to a.F" "inlining call to a.g\[go.shape.int\]"
}

// compile

// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// issue 8073.
// was "internal compiler error: overflow: float64 integer constant"

package main

func main() {
	var x int
	_ = float64(x * 0)
}

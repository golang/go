// compile

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package foo

type T struct {
	x int
	_ int
}

func main() {
	_ = T{0, 0}

	x := T{1, 1}
	_ = x
}

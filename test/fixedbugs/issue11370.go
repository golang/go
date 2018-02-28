// compile

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// issue 11370: cmd/compile: "0"[0] should not be a constant

package p

func main() {
	println(-"abc"[1] >> 1)
}

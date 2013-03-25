// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "./a"

var B = [2]string{"world", "hello"}

func main() {
	if a.A[0] != B[1] {
		panic("bad hello")
	}
}

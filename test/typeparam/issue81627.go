// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "cmp"

func main() {
	funcs := []func(int, int) bool{cmp.Less}
	if !funcs[0](1, 2) {
		panic("cmp.Less returned false")
	}
}

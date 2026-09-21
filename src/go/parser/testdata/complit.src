// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains examples of composite literals.
// The examples don't typecheck.

package p

var (
	_ = []int{}
	_ = [...]int{}
	_ = [42]int{}
	_ = map[string]int{}
	_ = T{}
	_ = T{1, 2, 3}
	_ = T{{}, T{}, []int{1, 2, 3}}
)

var (
	_ []int = {1, 2, 3}
	_ T = {1: {}, 2: {}}
	_ = T({"a", "b"})
	_ = f(x, y, []byte{1, 2, 3})
	_ = f(x, y, {1, 2, 3})
	_ = g({foo: x, bar: y})
)

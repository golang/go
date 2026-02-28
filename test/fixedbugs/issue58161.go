// compile -d=ssa/check/seed=1

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func F[G int]() int {
	return len(make([]int, copy([]G{}, []G{})))
}

func main() {
	F[int]()
}

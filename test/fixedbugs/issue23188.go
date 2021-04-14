// run

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test order of evaluation of index operations.

package main

func main() {
	arr := []int{1, 2}

	// The spec says that in an assignment statement the operands
	// of all index expressions and pointer indirections on the
	// left, and the expressions on the right, are evaluated in
	// the usual order. The usual order means function calls and
	// channel operations are done first. Then the assignments are
	// carried out one at a time. The operands of an index
	// expression include both the array and the index. So this
	// evaluates as
	//   tmp1 := arr
	//   tmp2 := len(arr) - 1
	//   tmp3 := len(arr)
	//   arr = arr[:tmp3-1]
	//   tmp1[tmp2] = 3
	arr, arr[len(arr)-1] = arr[:len(arr)-1], 3

	if len(arr) != 1 || arr[0] != 1 || arr[:2][1] != 3 {
		panic(arr)
	}
}

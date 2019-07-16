// compile

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Triggers a bug in writebarrier, which inserts one
// between (first block) OpAddr x and (second block) a VarDef x,
// which are then in the wrong order and unable to be
// properly scheduled.

package q

var S interface{}

func F(n int) {
	fun := func(x int) int {
		S = 1
		return n
	}
	i := fun(([]int{})[n])

	var fc [2]chan int
	S = (([1][2]chan int{fc})[i][i])
}

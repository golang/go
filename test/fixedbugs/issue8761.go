// compile

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// issue 8761
// used to confuse code generator into using temporary before initialization.
// caused 'variable live at entry' error in liveness analysis.

package p

func _() {
	type C chan int
	_ = [1][]C{[]C{make(chan int)}}
}

func _() {
	type C interface{}
	_ = [1][]C{[]C{recover()}}
}

func _() {
	type C *int
	_ = [1][]C{[]C{new(int)}}
}

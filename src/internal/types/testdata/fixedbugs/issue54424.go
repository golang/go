// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func f[P ~*T, T any]() {
	var p P
	var tp *T
	tp = p // this assignment is valid
	_ = tp
}

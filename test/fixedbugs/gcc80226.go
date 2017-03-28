// compile

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The gccgo compiler crashed while compiling a function that returned
// multiple zero-sized structs.
// https://gcc.gnu.org/PR80226.

package p

type S struct{}

func F() (S, S) {
	return S{}, S{}
}

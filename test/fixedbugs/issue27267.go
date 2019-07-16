// compile

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

// 1st test case from issue
type F = func(E) // compiles if not type alias or moved below E struct
type E struct {
	f F
}

var x = E{func(E) {}}

// 2nd test case from issue
type P = *S
type S struct {
	p P
}

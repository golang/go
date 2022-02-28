// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package b

import "a"

//go:noinline
func F() interface {} {
	return a.T[int]{}
}

//go:noinline
func G() interface{} {
	return struct{X,Y a.U}{}
}

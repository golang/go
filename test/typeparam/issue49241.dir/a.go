// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

type T[P any] struct {
	x P
}

type U struct {
	a,b int
}

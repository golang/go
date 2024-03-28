// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type innerT[P any, R *T0[P]] struct {
	Ref R
}

type T0[P any] struct {
	e innerT[P, *T0[P]]
}



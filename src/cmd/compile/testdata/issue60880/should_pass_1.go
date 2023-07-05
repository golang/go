// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type T0[P any] struct {
	e innerT[P, T0[P]]
}



type innerT[P any, R T0[P]] struct {
	Ref R
}

//Output:
// should_pass_1.go:7:6: invalid recursive type T0
//        should_pass_1.go:7:6: T0 refers to
//        should_pass_1.go:13:6: innerT refers to
//        should_pass_1.go:7:6: T0
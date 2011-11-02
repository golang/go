// $G $F.go || echo BUG: should compile

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type t1 int
type t2 int
type t3 int

func f1(t1, t2, t3)
func f2(t1, t2, t3 bool)
func f3(t1, t2, x t3)
func f4(t1, *t3)
func (x *t1) f5(y []t2) (t1, *t3)
func f6() (int, *string)
func f7(*t2, t3)
func f8(os int) int

func f9(os int) int {
	return os
}
func f10(err error) error {
	return err
}
func f11(t1 string) string {
	return t1
}

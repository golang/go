// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

type I interface {
	F()
}

type foo1 []byte
type foo2 []rune
type foo3 []uint8
type foo4 []int32
type foo5 string
type foo6 string
type foo7 string
type foo8 string
type foo9 string

func (f foo1) F() { return }
func (f foo2) F() { return }
func (f foo3) F() { return }
func (f foo4) F() { return }
func (f foo5) F() { return }
func (f foo6) F() { return }
func (f foo7) F() { return }
func (f foo8) F() { return }
func (f foo9) F() { return }

func Test1(s string) I  { return foo1(s) }
func Test2(s string) I  { return foo2(s) }
func Test3(s string) I  { return foo3(s) }
func Test4(s string) I  { return foo4(s) }
func Test5(s []byte) I  { return foo5(s) }
func Test6(s []rune) I  { return foo6(s) }
func Test7(s []uint8) I { return foo7(s) }
func Test8(s []int32) I { return foo8(s) }
func Test9(s int) I     { return foo9(s) }

type bar map[int]int

func (b bar) F() { return }

func TestBar() I { return bar{1: 2} }

type baz int

func IsBaz(x interface{}) bool { _, ok := x.(baz); return ok }

type baz2 int

func IsBaz2(x interface{}) bool {
	switch x.(type) {
	case baz2:
		return true
	default:
		return false
	}
}

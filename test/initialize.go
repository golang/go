// run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test initialization of package-level variables.

package main

import (
	"fmt"
	"reflect"
)

type S struct {
	A, B, C, X, Y, Z int
}

type T struct {
	S
}

var a1 = S{0, 0, 0, 1, 2, 3}
var b1 = S{X: 1, Z: 3, Y: 2}

var a2 = S{0, 0, 0, 0, 0, 0}
var b2 = S{}

var a3 = T{S{1, 2, 3, 0, 0, 0}}
var b3 = T{S: S{A: 1, B: 2, C: 3}}

var a4 = &[16]byte{0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0}
var b4 = &[16]byte{4: 1, 1, 1, 1, 12: 1, 1}

var a5 = &[16]byte{1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0}
var b5 = &[16]byte{1, 4: 1, 1, 1, 1, 12: 1, 1}

var a6 = &[16]byte{1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0}
var b6 = &[...]byte{1, 4: 1, 1, 1, 1, 12: 1, 1, 0, 0}

func f7(ch chan int) [2]chan int { return [2]chan int{ch, ch} }

var a7 = f7(make(chan int))

func f8(m map[string]string) [2]map[string]string { return [2]map[string]string{m, m} }
func m8(m [2]map[string]string) string {
	m[0]["def"] = "ghi"
	return m[1]["def"]
}

var a8 = f8(make(map[string]string))
var a9 = f8(map[string]string{"abc": "def"})

func f10(s *S) [2]*S { return [2]*S{s, s} }

var a10 = f10(new(S))
var a11 = f10(&S{X: 1})

func f12(b []byte) [2][]byte { return [2][]byte{b, b} }

var a12 = f12([]byte("hello"))
var a13 = f12([]byte{1, 2, 3})
var a14 = f12(make([]byte, 1))

func f15(b []rune) [2][]rune { return [2][]rune{b, b} }

var a15 = f15([]rune("hello"))
var a16 = f15([]rune{1, 2, 3})

type Same struct {
	a, b interface{}
}

var same = []Same{
	{a1, b1},
	{a2, b2},
	{a3, b3},
	{a4, b4},
	{a5, b5},
	{a6, b6},
	{a7[0] == a7[1], true},
	{m8(a8) == "ghi", true},
	{m8(a9) == "ghi", true},
	{a10[0] == a10[1], true},
	{a11[0] == a11[1], true},
	{&a12[0][0] == &a12[1][0], true},
	{&a13[0][0] == &a13[1][0], true},
	{&a14[0][0] == &a14[1][0], true},
	{&a15[0][0] == &a15[1][0], true},
	{&a16[0][0] == &a16[1][0], true},
}

func main() {
	ok := true
	for i, s := range same {
		if !reflect.DeepEqual(s.a, s.b) {
			ok = false
			fmt.Printf("#%d not same: %v and %v\n", i+1, s.a, s.b)
		}
	}
	if !ok {
		fmt.Println("BUG: test/initialize")
	}
}

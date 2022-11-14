// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package funcInference

import "strconv"

type any interface{}

func f0[A any, B interface{*C}, C interface{*D}, D interface{*A}](A, B, C, D) {}
func _() {
	f := f0[string]
	f("a", nil, nil, nil)
	f0("a", nil, nil, nil)
}

func f1[A any, B interface{*A}](A, B) {}
func _() {
	f := f1[int]
	f(int(0), new(int))
	f1(int(0), new(int))
}

func f2[A any, B interface{[]A}](A, B) {}
func _() {
	f := f2[byte]
	f(byte(0), []byte{})
	f2(byte(0), []byte{})
}

// Embedding stand-alone type parameters is not permitted for now. Disabled.
// func f3[A any, B interface{~C}, C interface{~*A}](A, B, C)
// func _() {
// 	f := f3[int]
// 	var x int
// 	f(x, &x, &x)
// 	f3(x, &x, &x)
// }

func f4[A any, B interface{[]C}, C interface{*A}](A, B, C) {}
func _() {
	f := f4[int]
	var x int
	f(x, []*int{}, &x)
	f4(x, []*int{}, &x)
}

func f5[A interface{struct{b B; c C}}, B any, C interface{*B}](x B) A { panic(0) }
func _() {
	x := f5(1.2)
	var _ float64 = x.b
	var _ float64 = *x.c
}

func f6[A any, B interface{~struct{f []A}}](B) A { panic(0) }
func _() {
	x := f6(struct{f []string}{})
	var _ string = x
}

func f7[A interface{*B}, B interface{~*A}]() {}

// More realistic examples

func Double[S interface{ ~[]E }, E interface{ ~int | ~int8 | ~int16 | ~int32 | ~int64 }](s S) S {
	r := make(S, len(s))
	for i, v := range s {
		r[i] = v + v
	}
	return r
}

type MySlice []int

var _ = Double(MySlice{1})

// From the draft design.

type Setter[B any] interface {
	Set(string)
	*B
}

func FromStrings[T interface{}, PT Setter[T]](s []string) []T {
	result := make([]T, len(s))
	for i, v := range s {
		// The type of &result[i] is *T which is in the type set
		// of Setter, so we can convert it to PT.
		p := PT(&result[i])
		// PT has a Set method.
		p.Set(v)
	}
	return result
}

type Settable int

func (p *Settable) Set(s string) {
	i, _ := strconv.Atoi(s) // real code should not ignore the error
	*p = Settable(i)
}

var _ = FromStrings[Settable]([]string{"1", "2"})

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Methods of reflect.rtype use StructOf and FuncOf which in turn depend on
// reflect.Value.Method. StructOf and FuncOf must not disable the DCE.

package main

import "reflect"

type S int

func (s S) M() { println("S.M") }

func (s S) N() { println("S.N") }

type T float64

func (t T) F(s S) {}

func useStructOf() {
	t := reflect.StructOf([]reflect.StructField{
		{
			Name: "X",
			Type: reflect.TypeOf(int(0)),
		},
	})
	println(t.Name())
}

func useFuncOf() {
	t := reflect.FuncOf(
		[]reflect.Type{reflect.TypeOf(int(0))},
		[]reflect.Type{reflect.TypeOf(int(0))},
		false,
	)
	println(t.Name())
}

func main() {
	useStructOf()
	useFuncOf()

	var t T
	meth, _ := reflect.TypeOf(t).MethodByName("F")
	ft := meth.Type
	at := ft.In(1)
	v := reflect.New(at).Elem()
	methV := v.MethodByName("M")
	methV.Call([]reflect.Value{v})
}

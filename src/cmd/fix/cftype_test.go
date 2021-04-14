// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func init() {
	addTestCases(cftypeTests, cftypefix)
}

var cftypeTests = []testCase{
	{
		Name: "cftype.localVariable",
		In: `package main

import "C"

func f() {
	var x C.CFTypeRef = nil
	x = nil
	x, x = nil, nil
}
`,
		Out: `package main

import "C"

func f() {
	var x C.CFTypeRef = 0
	x = 0
	x, x = 0, 0
}
`,
	},
	{
		Name: "cftype.globalVariable",
		In: `package main

import "C"

var x C.CFTypeRef = nil

func f() {
	x = nil
}
`,
		Out: `package main

import "C"

var x C.CFTypeRef = 0

func f() {
	x = 0
}
`,
	},
	{
		Name: "cftype.EqualArgument",
		In: `package main

import "C"

var x C.CFTypeRef
var y = x == nil
var z = x != nil
`,
		Out: `package main

import "C"

var x C.CFTypeRef
var y = x == 0
var z = x != 0
`,
	},
	{
		Name: "cftype.StructField",
		In: `package main

import "C"

type T struct {
	x C.CFTypeRef
}

var t = T{x: nil}
`,
		Out: `package main

import "C"

type T struct {
	x C.CFTypeRef
}

var t = T{x: 0}
`,
	},
	{
		Name: "cftype.FunctionArgument",
		In: `package main

import "C"

func f(x C.CFTypeRef) {
}

func g() {
	f(nil)
}
`,
		Out: `package main

import "C"

func f(x C.CFTypeRef) {
}

func g() {
	f(0)
}
`,
	},
	{
		Name: "cftype.ArrayElement",
		In: `package main

import "C"

var x = [3]C.CFTypeRef{nil, nil, nil}
`,
		Out: `package main

import "C"

var x = [3]C.CFTypeRef{0, 0, 0}
`,
	},
	{
		Name: "cftype.SliceElement",
		In: `package main

import "C"

var x = []C.CFTypeRef{nil, nil, nil}
`,
		Out: `package main

import "C"

var x = []C.CFTypeRef{0, 0, 0}
`,
	},
	{
		Name: "cftype.MapKey",
		In: `package main

import "C"

var x = map[C.CFTypeRef]int{nil: 0}
`,
		Out: `package main

import "C"

var x = map[C.CFTypeRef]int{0: 0}
`,
	},
	{
		Name: "cftype.MapValue",
		In: `package main

import "C"

var x = map[int]C.CFTypeRef{0: nil}
`,
		Out: `package main

import "C"

var x = map[int]C.CFTypeRef{0: 0}
`,
	},
	{
		Name: "cftype.Conversion1",
		In: `package main

import "C"

var x C.CFTypeRef
var y = (*unsafe.Pointer)(&x)
`,
		Out: `package main

import "C"

var x C.CFTypeRef
var y = (*unsafe.Pointer)(unsafe.Pointer(&x))
`,
	},
	{
		Name: "cftype.Conversion2",
		In: `package main

import "C"

var x unsafe.Pointer
var y = (*C.CFTypeRef)(&x)
`,
		Out: `package main

import "C"

var x unsafe.Pointer
var y = (*C.CFTypeRef)(unsafe.Pointer(&x))
`,
	},
}

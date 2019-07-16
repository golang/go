// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func init() {
	addTestCases(eglTests, eglfix)
}

var eglTests = []testCase{
	{
		Name: "egl.localVariable",
		In: `package main

import "C"

func f() {
	var x C.EGLDisplay = nil
	x = nil
	x, x = nil, nil
}
`,
		Out: `package main

import "C"

func f() {
	var x C.EGLDisplay = 0
	x = 0
	x, x = 0, 0
}
`,
	},
	{
		Name: "egl.globalVariable",
		In: `package main

import "C"

var x C.EGLDisplay = nil

func f() {
	x = nil
}
`,
		Out: `package main

import "C"

var x C.EGLDisplay = 0

func f() {
	x = 0
}
`,
	},
	{
		Name: "egl.EqualArgument",
		In: `package main

import "C"

var x C.EGLDisplay
var y = x == nil
var z = x != nil
`,
		Out: `package main

import "C"

var x C.EGLDisplay
var y = x == 0
var z = x != 0
`,
	},
	{
		Name: "egl.StructField",
		In: `package main

import "C"

type T struct {
	x C.EGLDisplay
}

var t = T{x: nil}
`,
		Out: `package main

import "C"

type T struct {
	x C.EGLDisplay
}

var t = T{x: 0}
`,
	},
	{
		Name: "egl.FunctionArgument",
		In: `package main

import "C"

func f(x C.EGLDisplay) {
}

func g() {
	f(nil)
}
`,
		Out: `package main

import "C"

func f(x C.EGLDisplay) {
}

func g() {
	f(0)
}
`,
	},
	{
		Name: "egl.ArrayElement",
		In: `package main

import "C"

var x = [3]C.EGLDisplay{nil, nil, nil}
`,
		Out: `package main

import "C"

var x = [3]C.EGLDisplay{0, 0, 0}
`,
	},
	{
		Name: "egl.SliceElement",
		In: `package main

import "C"

var x = []C.EGLDisplay{nil, nil, nil}
`,
		Out: `package main

import "C"

var x = []C.EGLDisplay{0, 0, 0}
`,
	},
	{
		Name: "egl.MapKey",
		In: `package main

import "C"

var x = map[C.EGLDisplay]int{nil: 0}
`,
		Out: `package main

import "C"

var x = map[C.EGLDisplay]int{0: 0}
`,
	},
	{
		Name: "egl.MapValue",
		In: `package main

import "C"

var x = map[int]C.EGLDisplay{0: nil}
`,
		Out: `package main

import "C"

var x = map[int]C.EGLDisplay{0: 0}
`,
	},
}

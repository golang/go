// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func init() {
	addTestCases(jniTests, jnifix)
}

var jniTests = []testCase{
	{
		Name: "jni.localVariable",
		In: `package main

import "C"

func f() {
	var x C.jobject = nil
	x = nil
	x, x = nil, nil
}
`,
		Out: `package main

import "C"

func f() {
	var x C.jobject = 0
	x = 0
	x, x = 0, 0
}
`,
	},
	{
		Name: "jni.globalVariable",
		In: `package main

import "C"

var x C.jobject = nil

func f() {
	x = nil
}
`,
		Out: `package main

import "C"

var x C.jobject = 0

func f() {
	x = 0
}
`,
	},
	{
		Name: "jni.EqualArgument",
		In: `package main

import "C"

var x C.jobject
var y = x == nil
var z = x != nil
`,
		Out: `package main

import "C"

var x C.jobject
var y = x == 0
var z = x != 0
`,
	},
	{
		Name: "jni.StructField",
		In: `package main

import "C"

type T struct {
	x C.jobject
}

var t = T{x: nil}
`,
		Out: `package main

import "C"

type T struct {
	x C.jobject
}

var t = T{x: 0}
`,
	},
	{
		Name: "jni.FunctionArgument",
		In: `package main

import "C"

func f(x C.jobject) {
}

func g() {
	f(nil)
}
`,
		Out: `package main

import "C"

func f(x C.jobject) {
}

func g() {
	f(0)
}
`,
	},
	{
		Name: "jni.ArrayElement",
		In: `package main

import "C"

var x = [3]C.jobject{nil, nil, nil}
`,
		Out: `package main

import "C"

var x = [3]C.jobject{0, 0, 0}
`,
	},
	{
		Name: "jni.SliceElement",
		In: `package main

import "C"

var x = []C.jobject{nil, nil, nil}
`,
		Out: `package main

import "C"

var x = []C.jobject{0, 0, 0}
`,
	},
	{
		Name: "jni.MapKey",
		In: `package main

import "C"

var x = map[C.jobject]int{nil: 0}
`,
		Out: `package main

import "C"

var x = map[C.jobject]int{0: 0}
`,
	},
	{
		Name: "jni.MapValue",
		In: `package main

import "C"

var x = map[int]C.jobject{0: nil}
`,
		Out: `package main

import "C"

var x = map[int]C.jobject{0: 0}
`,
	},
}

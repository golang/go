// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "strings"

func init() {
	addTestCases(eglTestsFor("EGLDisplay"), eglfixDisp)
	addTestCases(eglTestsFor("EGLConfig"), eglfixConfig)
}

func eglTestsFor(tname string) []testCase {
	var eglTests = []testCase{
		{
			Name: "egl.localVariable",
			In: `package main

// typedef void *$EGLTYPE;
import "C"

func f() {
	var x C.$EGLTYPE = nil
	x = nil
	x, x = nil, nil
}
`,
			Out: `package main

// typedef void *$EGLTYPE;
import "C"

func f() {
	var x C.$EGLTYPE = 0
	x = 0
	x, x = 0, 0
}
`,
		},
		{
			Name: "egl.globalVariable",
			In: `package main

// typedef void *$EGLTYPE;
import "C"

var x C.$EGLTYPE = nil

func f() {
	x = nil
}
`,
			Out: `package main

// typedef void *$EGLTYPE;
import "C"

var x C.$EGLTYPE = 0

func f() {
	x = 0
}
`,
		},
		{
			Name: "egl.EqualArgument",
			In: `package main

// typedef void *$EGLTYPE;
import "C"

var x C.$EGLTYPE
var y = x == nil
var z = x != nil
`,
			Out: `package main

// typedef void *$EGLTYPE;
import "C"

var x C.$EGLTYPE
var y = x == 0
var z = x != 0
`,
		},
		{
			Name: "egl.StructField",
			In: `package main

// typedef void *$EGLTYPE;
import "C"

type T struct {
	x C.$EGLTYPE
}

var t = T{x: nil}
`,
			Out: `package main

// typedef void *$EGLTYPE;
import "C"

type T struct {
	x C.$EGLTYPE
}

var t = T{x: 0}
`,
		},
		{
			Name: "egl.FunctionArgument",
			In: `package main

// typedef void *$EGLTYPE;
import "C"

func f(x C.$EGLTYPE) {
}

func g() {
	f(nil)
}
`,
			Out: `package main

// typedef void *$EGLTYPE;
import "C"

func f(x C.$EGLTYPE) {
}

func g() {
	f(0)
}
`,
		},
		{
			Name: "egl.ArrayElement",
			In: `package main

// typedef void *$EGLTYPE;
import "C"

var x = [3]C.$EGLTYPE{nil, nil, nil}
`,
			Out: `package main

// typedef void *$EGLTYPE;
import "C"

var x = [3]C.$EGLTYPE{0, 0, 0}
`,
		},
		{
			Name: "egl.SliceElement",
			In: `package main

// typedef void *$EGLTYPE;
import "C"

var x = []C.$EGLTYPE{nil, nil, nil}
`,
			Out: `package main

// typedef void *$EGLTYPE;
import "C"

var x = []C.$EGLTYPE{0, 0, 0}
`,
		},
		{
			Name: "egl.MapKey",
			In: `package main

// typedef void *$EGLTYPE;
import "C"

var x = map[C.$EGLTYPE]int{nil: 0}
`,
			Out: `package main

// typedef void *$EGLTYPE;
import "C"

var x = map[C.$EGLTYPE]int{0: 0}
`,
		},
		{
			Name: "egl.MapValue",
			In: `package main

// typedef void *$EGLTYPE;
import "C"

var x = map[int]C.$EGLTYPE{0: nil}
`,
			Out: `package main

// typedef void *$EGLTYPE;
import "C"

var x = map[int]C.$EGLTYPE{0: 0}
`,
		},
	}
	for i := range eglTests {
		t := &eglTests[i]
		t.In = strings.ReplaceAll(t.In, "$EGLTYPE", tname)
		t.Out = strings.ReplaceAll(t.Out, "$EGLTYPE", tname)
	}
	return eglTests
}

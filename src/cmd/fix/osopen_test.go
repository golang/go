// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func init() {
	addTestCases(osopenTests, osopen)
}

var osopenTests = []testCase{
	{
		Name: "osopen.0",
		In: `package main

import (
	"os"
)

func f() {
	os.OpenFile(a, b, c)
	os.Open(a, os.O_RDONLY, 0)
	os.Open(a, os.O_RDONLY, 0666)
	os.Open(a, os.O_RDWR, 0)
	os.Open(a, os.O_CREAT, 0666)
	os.Open(a, os.O_CREAT|os.O_TRUNC, 0664)
	os.Open(a, os.O_CREATE, 0666)
	os.Open(a, os.O_CREATE|os.O_TRUNC, 0664)
	os.Open(a, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0666)
	os.Open(a, os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0666)
	os.Open(a, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0666)
	os.Open(a, os.O_SURPRISE|os.O_CREATE, 0666)
	_ = os.O_CREAT
}
`,
		Out: `package main

import (
	"os"
)

func f() {
	os.OpenFile(a, b, c)
	os.Open(a)
	os.Open(a)
	os.OpenFile(a, os.O_RDWR, 0)
	os.Create(a)
	os.Create(a)
	os.Create(a)
	os.Create(a)
	os.Create(a)
	os.OpenFile(a, os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0666)
	os.OpenFile(a, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0666)
	os.OpenFile(a, os.O_SURPRISE|os.O_CREATE, 0666)
	_ = os.O_CREATE
}
`,
	},
	{
		Name: "osopen.1",
		In: `package main

import (
	"os"
)

func f() {
	_ = os.O_CREAT
}
`,
		Out: `package main

import (
	"os"
)

func f() {
	_ = os.O_CREATE
}
`,
	},
}

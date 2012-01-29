// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func init() {
	addTestCases(oserrorstringTests, oserrorstring)
}

var oserrorstringTests = []testCase{
	{
		Name: "oserrorstring.0",
		In: `package main

import "os"

var _ = os.ErrorString("foo")
var _ os.Error = os.ErrorString("bar1")
var _ os.Error = os.NewError("bar2")
var _ os.Error = MyError("bal") // don't rewrite this one

var (
	_          = os.ErrorString("foo")
	_ os.Error = os.ErrorString("bar1")
	_ os.Error = os.NewError("bar2")
	_ os.Error = MyError("bal") // don't rewrite this one
)

func _() (err os.Error) {
	err = os.ErrorString("foo")
	return os.ErrorString("foo")
}
`,
		Out: `package main

import "os"

var _ = os.NewError("foo")
var _ = os.NewError("bar1")
var _ = os.NewError("bar2")
var _ os.Error = MyError("bal") // don't rewrite this one

var (
	_          = os.NewError("foo")
	_          = os.NewError("bar1")
	_          = os.NewError("bar2")
	_ os.Error = MyError("bal") // don't rewrite this one
)

func _() (err os.Error) {
	err = os.NewError("foo")
	return os.NewError("foo")
}
`,
	},
}

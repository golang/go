// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func init() {
	addTestCases(oswaitTests, oswait)
}

var oswaitTests = []testCase{
	{
		Name: "oswait.0",
		In: `package main

import (
	"os"
)

func f() {
	os.Wait()
	os.Wait(0)
	os.Wait(1)
	os.Wait(A | B)
}
`,
		Out: `package main

import (
	"os"
)

func f() {
	os.Wait()
	os.Wait()
	os.Wait(1)
	os.Wait(A | B)
}
`,
	},
}

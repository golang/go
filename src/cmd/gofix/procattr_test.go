// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func init() {
	addTestCases(procattrTests, procattr)
}

var procattrTests = []testCase{
	{
		Name: "procattr.0",
		In: `package main

import (
	"os"
	"syscall"
)

func f() {
	os.StartProcess(a, b, c, d, e)
	os.StartProcess(a, b, os.Environ(), d, e)
	os.StartProcess(a, b, nil, d, e)
	os.StartProcess(a, b, c, "", e)
	os.StartProcess(a, b, c, d, nil)
	os.StartProcess(a, b, nil, "", nil)

	os.StartProcess(
		a,
		b,
		c,
		d,
		e,
	)

	syscall.StartProcess(a, b, c, d, e)
	syscall.StartProcess(a, b, os.Environ(), d, e)
	syscall.StartProcess(a, b, nil, d, e)
	syscall.StartProcess(a, b, c, "", e)
	syscall.StartProcess(a, b, c, d, nil)
	syscall.StartProcess(a, b, nil, "", nil)
}
`,
		Out: `package main

import (
	"os"
	"syscall"
)

func f() {
	os.StartProcess(a, b, &os.ProcAttr{Env: c, Dir: d, Files: e})
	os.StartProcess(a, b, &os.ProcAttr{Dir: d, Files: e})
	os.StartProcess(a, b, &os.ProcAttr{Dir: d, Files: e})
	os.StartProcess(a, b, &os.ProcAttr{Env: c, Files: e})
	os.StartProcess(a, b, &os.ProcAttr{Env: c, Dir: d})
	os.StartProcess(a, b, &os.ProcAttr{})

	os.StartProcess(
		a,
		b, &os.ProcAttr{Env: c, Dir: d, Files: e},
	)

	syscall.StartProcess(a, b, &syscall.ProcAttr{Env: c, Dir: d, Files: e})
	syscall.StartProcess(a, b, &syscall.ProcAttr{Dir: d, Files: e})
	syscall.StartProcess(a, b, &syscall.ProcAttr{Dir: d, Files: e})
	syscall.StartProcess(a, b, &syscall.ProcAttr{Env: c, Files: e})
	syscall.StartProcess(a, b, &syscall.ProcAttr{Env: c, Dir: d})
	syscall.StartProcess(a, b, &syscall.ProcAttr{})
}
`,
	},
}

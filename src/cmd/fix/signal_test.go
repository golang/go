// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func init() {
	addTestCases(signalTests, signal)
}

var signalTests = []testCase{
	{
		Name: "signal.0",
		In: `package main

import (
	_ "a"
	"os/signal"
	_ "z"
)

type T1 signal.UnixSignal
type T2 signal.Signal

func f() {
	_ = signal.SIGHUP
	_ = signal.Incoming
}
`,
		Out: `package main

import (
	_ "a"
	"os"
	"os/signal"
	_ "z"
)

type T1 os.UnixSignal
type T2 os.Signal

func f() {
	_ = os.SIGHUP
	_ = signal.Incoming
}
`,
	},
	{
		Name: "signal.1",
		In: `package main

import (
	"os"
	"os/signal"
)

func f() {
	var _ os.Error
	_ = signal.SIGHUP
}
`,
		Out: `package main

import "os"

func f() {
	var _ os.Error
	_ = os.SIGHUP
}
`,
	},
	{
		Name: "signal.2",
		In: `package main

import "os"
import "os/signal"

func f() {
	var _ os.Error
	_ = signal.SIGHUP
}
`,
		Out: `package main

import "os"

func f() {
	var _ os.Error
	_ = os.SIGHUP
}
`,
	},
}

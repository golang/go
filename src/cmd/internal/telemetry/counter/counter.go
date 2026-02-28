// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !cmd_go_bootstrap && !compiler_bootstrap

package counter

import (
	"flag"
	"os"

	"golang.org/x/telemetry/counter"
)

var openCalled bool

func OpenCalled() bool { return openCalled }

// Open opens the counter files for writing if telemetry is supported
// on the current platform (and does nothing otherwise).
func Open() {
	openCalled = true
	counter.OpenDir(os.Getenv("TEST_TELEMETRY_DIR"))
}

// Inc increments the counter with the given name.
func Inc(name string) {
	counter.Inc(name)
}

// New returns a counter with the given name.
func New(name string) *counter.Counter {
	return counter.New(name)
}

// NewStack returns a new stack counter with the given name and depth.
func NewStack(name string, depth int) *counter.StackCounter {
	return counter.NewStack(name, depth)
}

// CountFlags creates a counter for every flag that is set
// and increments the counter. The name of the counter is
// the concatenation of prefix and the flag name.
func CountFlags(prefix string, flagSet flag.FlagSet) {
	counter.CountFlags(prefix, flagSet)
}

// CountFlagValue creates a counter for the flag value
// if it is set and increments the counter. The name of the
// counter is the concatenation of prefix, the flagName, ":",
// and value.String() for the flag's value.
func CountFlagValue(prefix string, flagSet flag.FlagSet, flagName string) {
	// TODO(matloob): Maybe pass in a list of flagNames if we end up counting
	// values for more than one?
	// TODO(matloob): Add this to x/telemetry?
	flagSet.Visit(func(f *flag.Flag) {
		if f.Name == flagName {
			counter.New(prefix + f.Name + ":" + f.Value.String()).Inc()
		}
	})
}

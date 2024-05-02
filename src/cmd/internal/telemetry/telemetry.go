// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !cmd_go_bootstrap && !compiler_bootstrap

// Package telemetry is a shim package around the golang.org/x/telemetry
// and golang.org/x/telemetry/counter packages that has code build tagged
// out for cmd_go_bootstrap so that the bootstrap Go command does not
// depend on net (which is a dependency of golang.org/x/telemetry/counter
// on Windows).
package telemetry

import (
	"flag"
	"os"

	"golang.org/x/telemetry"
	"golang.org/x/telemetry/counter"
)

// Start opens the counter files for writing if telemetry is supported
// on the current platform (and does nothing otherwise).
func Start() {
	telemetry.Start(telemetry.Config{
		TelemetryDir: os.Getenv("TEST_TELEMETRY_DIR"),
	})
}

// StartWithUpload opens the counter files for writing if telemetry
// is supported on the current platform and also enables a once a day
// check to see if the weekly reports are ready to be uploaded.
// It should only be called by cmd/go
func StartWithUpload() {
	telemetry.Start(telemetry.Config{
		Upload:       true,
		TelemetryDir: os.Getenv("TEST_TELEMETRY_DIR"),
	})
}

// Inc increments the counter with the given name.
func Inc(name string) {
	counter.Inc(name)
}

// NewCounter returns a counter with the given name.
func NewCounter(name string) *counter.Counter {
	return counter.New(name)
}

// NewStack returns a new stack counter with the given name and depth.
func NewStackCounter(name string, depth int) *counter.StackCounter {
	return counter.NewStack(name, depth)
}

// CountFlags creates a counter for every flag that is set
// and increments the counter. The name of the counter is
// the concatenation of prefix and the flag name.
func CountFlags(prefix string, flagSet flag.FlagSet) {
	counter.CountFlags(prefix, flagSet)
}

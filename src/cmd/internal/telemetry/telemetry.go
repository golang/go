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

// Mode returns the current telemetry mode.
//
// The telemetry mode is a global value that controls both the local collection
// and uploading of telemetry data. Possible mode values are:
//   - "on":    both collection and uploading is enabled
//   - "local": collection is enabled, but uploading is disabled
//   - "off":   both collection and uploading are disabled
//
// When mode is "on", or "local", telemetry data is written to the local file
// system and may be inspected with the [gotelemetry] command.
//
// If an error occurs while reading the telemetry mode from the file system,
// Mode returns the default value "local".
//
// [gotelemetry]: https://pkg.go.dev/golang.org/x/telemetry/cmd/gotelemetry
func Mode() string {
	return telemetry.Mode()
}

// SetMode sets the global telemetry mode to the given value.
//
// See the documentation of [Mode] for a description of the supported mode
// values.
//
// An error is returned if the provided mode value is invalid, or if an error
// occurs while persisting the mode value to the file system.
func SetMode(mode string) error {
	return telemetry.SetMode(mode)
}

// Dir returns the telemetry directory.
func Dir() string {
	return telemetry.Dir()
}

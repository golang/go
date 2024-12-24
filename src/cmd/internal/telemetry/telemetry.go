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
	"os"

	"cmd/internal/telemetry/counter"

	"golang.org/x/telemetry"
)

var openCountersCalled, maybeChildCalled bool

// MaybeParent does a once a day check to see if the weekly reports are
// ready to be processed or uploaded, and if so, starts the telemetry child to
// do so. It should only be called by cmd/go, and only after OpenCounters and MaybeChild
// have already been called.
func MaybeParent() {
	if !counter.OpenCalled() || !maybeChildCalled {
		panic("MaybeParent must be called after OpenCounters and MaybeChild")
	}
	telemetry.Start(telemetry.Config{
		Upload:       true,
		TelemetryDir: os.Getenv("TEST_TELEMETRY_DIR"),
	})
}

// MaybeChild executes the telemetry child logic if the calling program is
// the telemetry child process, and does nothing otherwise. It is meant to be
// called as the first thing in a program that uses telemetry.OpenCounters but cannot
// call telemetry.OpenCounters immediately when it starts.
func MaybeChild() {
	maybeChildCalled = true
	telemetry.MaybeChild(telemetry.Config{
		Upload:       true,
		TelemetryDir: os.Getenv("TEST_TELEMETRY_DIR"),
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

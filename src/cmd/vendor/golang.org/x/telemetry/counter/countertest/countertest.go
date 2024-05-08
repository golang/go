// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// countertest provides testing utilities for counters.
// This package cannot be used except for testing.
package countertest

import (
	"sync"

	"golang.org/x/telemetry/counter"
	ic "golang.org/x/telemetry/internal/counter"
	"golang.org/x/telemetry/internal/telemetry"
)

var (
	openedMu sync.Mutex
	opened   bool
)

// SupportedPlatform reports if this platform supports Open()
const SupportedPlatform = !telemetry.DisabledOnPlatform

func isOpen() bool {
	openedMu.Lock()
	defer openedMu.Unlock()
	return opened
}

// Open enables telemetry data writing to disk.
// This is supposed to be called once during the program execution
// (i.e. typically in TestMain), and must not be used with
// golang.org/x/telemetry/counter.Open.
func Open(telemetryDir string) {
	openedMu.Lock()
	defer openedMu.Unlock()
	if opened {
		panic("Open was called more than once")
	}
	telemetry.Default = telemetry.NewDir(telemetryDir)

	counter.Open()
	opened = true
}

// ReadCounter reads the given counter.
func ReadCounter(c *counter.Counter) (count uint64, _ error) {
	return ic.Read(c)
}

// ReadStackCounter reads the given StackCounter.
func ReadStackCounter(c *counter.StackCounter) (stackCounts map[string]uint64, _ error) {
	return ic.ReadStack(c)
}

// ReadFile reads the counters and stack counters from the given file.
func ReadFile(name string) (counters, stackCounters map[string]uint64, _ error) {
	return ic.ReadFile(name)
}

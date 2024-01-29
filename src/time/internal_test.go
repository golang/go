// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package time

func init() {
	// Force US/Pacific for time zone tests.
	ForceUSPacificForTesting()
}

func initTestingZone() {
	// For hermeticity, use only tzinfo source from the test's GOROOT,
	// not the system sources and not whatever GOROOT may happen to be
	// set in the process's environment (if any).
	// This test runs in GOROOT/src/time, so GOROOT is "../..",
	// but it is theoretically possible
	sources := []string{"../../lib/time/zoneinfo.zip"}
	z, err := loadLocation("America/Los_Angeles", sources)
	if err != nil {
		panic("cannot load America/Los_Angeles for testing: " + err.Error() + "; you may want to use -tags=timetzdata")
	}
	z.name = "Local"
	localLoc = *z
}

var origPlatformZoneSources []string = platformZoneSources

func disablePlatformSources() (undo func()) {
	platformZoneSources = nil
	return func() {
		platformZoneSources = origPlatformZoneSources
	}
}

var Interrupt = interrupt
var DaysIn = daysIn

func empty(arg any, seq uintptr) {}

// Test that a runtimeTimer with a period that would overflow when on
// expiration does not throw or cause other timers to hang.
//
// This test has to be in internal_test.go since it fiddles with
// unexported data structures.
func CheckRuntimeTimerPeriodOverflow() {
	// We manually create a runtimeTimer with huge period, but that expires
	// immediately. The public Timer interface would require waiting for
	// the entire period before the first update.
	r := &runtimeTimer{
		When:   runtimeNano(),
		Period: 1<<63 - 1,
		F:      empty,
		Arg:    nil,
	}
	startTimer(r)
	defer stopTimer(r)

	// If this test fails, we will either throw (when siftdownTimer detects
	// bad when on update), or other timers will hang (if the timer in a
	// heap is in a bad state). There is no reliable way to test this, but
	// we wait on a short timer here as a smoke test (alternatively, timers
	// in later tests may hang).
	<-After(25 * Millisecond)
}

var (
	MinMonoTime = Time{wall: 1 << 63, ext: -1 << 63, loc: UTC}
	MaxMonoTime = Time{wall: 1 << 63, ext: 1<<63 - 1, loc: UTC}

	NotMonoNegativeTime = Time{wall: 0, ext: -1<<63 + 50}
)

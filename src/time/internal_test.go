// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package time

func init() {
	// force US/Pacific for time zone tests
	ForceUSPacificForTesting()
}

func initTestingZone() {
	z, err := loadLocation("America/Los_Angeles", zoneSources[len(zoneSources)-1:])
	if err != nil {
		panic("cannot load America/Los_Angeles for testing: " + err.Error())
	}
	z.name = "Local"
	localLoc = *z
}

var OrigZoneSources = zoneSources

func forceZipFileForTesting(zipOnly bool) {
	zoneSources = make([]string, len(OrigZoneSources))
	copy(zoneSources, OrigZoneSources)
	if zipOnly {
		zoneSources = zoneSources[len(zoneSources)-1:]
	}
}

var Interrupt = interrupt
var DaysIn = daysIn

func empty(arg interface{}, seq uintptr) {}

// Test that a runtimeTimer with a duration so large it overflows
// does not cause other timers to hang.
//
// This test has to be in internal_test.go since it fiddles with
// unexported data structures.
func CheckRuntimeTimerOverflow() {
	// We manually create a runtimeTimer to bypass the overflow
	// detection logic in NewTimer: we're testing the underlying
	// runtime.addtimer function.
	r := &runtimeTimer{
		when: runtimeNano() + (1<<63 - 1),
		f:    empty,
		arg:  nil,
	}
	startTimer(r)

	// Start a goroutine that should send on t.C right away.
	t := NewTimer(1)

	defer func() {
		// Subsequent tests won't work correctly if we don't stop the
		// overflow timer and kick the timer proc back into service.
		//
		// The timer proc is now sleeping and can only be awoken by
		// adding a timer to the *beginning* of the heap. We can't
		// wake it up by calling NewTimer since other tests may have
		// left timers running that should have expired before ours.
		// Instead we zero the overflow timer duration and start it
		// once more.
		stopTimer(r)
		t.Stop()
		resetTimer(r, 0)
	}()

	// If the test fails, we will hang here until the timeout in the
	// testing package fires, which is 10 minutes. It would be nice to
	// catch the problem sooner, but there is no reliable way to guarantee
	// that timers are run without doing something involving the scheduler.
	// Previous failed attempts have tried calling runtime.Gosched and
	// runtime.GC, but neither is reliable. So we fall back to hope:
	// We hope we don't hang here.
	<-t.C
}

var (
	MinMonoTime = Time{wall: 1 << 63, ext: -1 << 63, loc: UTC}
	MaxMonoTime = Time{wall: 1 << 63, ext: 1<<63 - 1, loc: UTC}
)

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Export guts for testing.

package runtime

const SiginfoMaxSize = _si_max_size
const SigeventMaxSize = _sigev_max_size

var NewOSProc0 = newosproc0
var Mincore = mincore
var ParseRelease = parseRelease
var ParseProcStatmRSS = parseProcStatmRSS

type Siginfo siginfo
type Sigevent sigevent

// TraceFailedPreemptions sends two preemption signals to the current M.
// The held M lock makes the interrupted goroutine unsafe to preempt.
// It returns a zero goroutine ID if asynchronous preemption is disabled.
//
//go:noinline
func TraceFailedPreemptions() (goid uint64, ok bool) {
	if debug.asyncpreemptoff != 0 {
		return 0, false
	}
	gp := getg()
	mp := acquirem()
	defer releasem(mp)

	deadline := nanotime() + 1e9
	for range 2 {
		for !castogscanstatus(gp, _Grunning, _Gscanrunning) {
			if nanotime() >= deadline {
				return gp.goid, false
			}
			procyield(10)
		}
		gp.preempt = true
		gen := mp.preemptGen.Load()
		preemptM(gp)
		for mp.preemptGen.Load() == gen {
			if nanotime() >= deadline {
				return gp.goid, false
			}
			procyield(10)
		}
	}
	return gp.goid, true
}

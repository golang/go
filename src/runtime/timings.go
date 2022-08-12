// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"runtime/internal/atomic"
)

// Check all Ps and returns the time when the next timer should fire.
// Returns maxWhen if there are no timers.
// This is only called by sysmon and checkdead.
func timeSleepUntil() int64 {
	next := int64(maxWhen)

	// Prevent allp slice changes. This is like retake.
	lock(&allpLock)
	for _, pp := range allp {
		if pp == nil {
			// This can happen if procresize has grown
			// allp but not yet created new Ps.
			continue
		}

		w := int64(atomic.Load64(&pp.timing.timer0When))
		if w != 0 && w < next {
			next = w
		}

		w = int64(atomic.Load64(&pp.timing.timerModifiedEarliest))
		if w != 0 && w < next {
			next = w
		}
	}
	unlock(&allpLock)

	return next
}

// Check all Ps for a timer expiring sooner than pollUntil.
//
// Returns updated pollUntil value.
func checkTimersNoP(allpSnapshot []*p, timerpMaskSnapshot pMask, pollUntil int64) int64 {
	for id, p2 := range allpSnapshot {
		if timerpMaskSnapshot.read(uint32(id)) {
			w := p2.timing.noBarrierWakeTime()
			if w != 0 && (pollUntil == 0 || w < pollUntil) {
				pollUntil = w
			}
		}
	}

	return pollUntil
}

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package quic

import (
	"time"
)

type rttState struct {
	minRTT          time.Duration
	latestRTT       time.Duration
	smoothedRTT     time.Duration
	rttvar          time.Duration // RTT variation
	firstSampleTime time.Time     // time of first RTT sample
}

func (r *rttState) init() {
	r.minRTT = -1 // -1 indicates the first sample has not been taken yet

	// "[...] the initial RTT SHOULD be set to 333 milliseconds."
	// https://www.rfc-editor.org/rfc/rfc9002.html#section-6.2.2-1
	const initialRTT = 333 * time.Millisecond

	// https://www.rfc-editor.org/rfc/rfc9002.html#section-5.3-12
	r.smoothedRTT = initialRTT
	r.rttvar = initialRTT / 2
}

func (r *rttState) establishPersistentCongestion() {
	// "Endpoints SHOULD set the min_rtt to the newest RTT sample
	// after persistent congestion is established."
	// https://www.rfc-editor.org/rfc/rfc9002#section-5.2-5
	r.minRTT = r.latestRTT
}

// updateSample is called when we generate a new RTT sample.
// https://www.rfc-editor.org/rfc/rfc9002.html#section-5
func (r *rttState) updateSample(now time.Time, handshakeConfirmed bool, spaceID numberSpace, latestRTT, ackDelay, maxAckDelay time.Duration) {
	r.latestRTT = latestRTT

	if r.minRTT < 0 {
		// First RTT sample.
		// "min_rtt MUST be set to the latest_rtt on the first RTT sample."
		// https://www.rfc-editor.org/rfc/rfc9002.html#section-5.2-2
		r.minRTT = latestRTT
		// https://www.rfc-editor.org/rfc/rfc9002.html#section-5.3-14
		r.smoothedRTT = latestRTT
		r.rttvar = latestRTT / 2
		r.firstSampleTime = now
		return
	}

	// "min_rtt MUST be set to the lesser of min_rtt and latest_rtt [...]
	// on all other samples."
	// https://www.rfc-editor.org/rfc/rfc9002.html#section-5.2-2
	r.minRTT = min(r.minRTT, latestRTT)

	// https://www.rfc-editor.org/rfc/rfc9002.html#section-5.3-16
	if handshakeConfirmed {
		ackDelay = min(ackDelay, maxAckDelay)
	}
	adjustedRTT := latestRTT - ackDelay
	if adjustedRTT < r.minRTT {
		adjustedRTT = latestRTT
	}
	rttvarSample := abs(r.smoothedRTT - adjustedRTT)
	r.rttvar = (3*r.rttvar + rttvarSample) / 4
	r.smoothedRTT = ((7 * r.smoothedRTT) + adjustedRTT) / 8
}

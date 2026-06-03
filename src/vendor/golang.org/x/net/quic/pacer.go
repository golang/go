// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package quic

import (
	"time"
)

// A pacerState controls the rate at which packets are sent using a leaky-bucket rate limiter.
//
// The pacer limits the maximum size of a burst of packets.
// When a burst exceeds this limit, it spreads subsequent packets
// over time.
//
// The bucket is initialized to the maximum burst size (ten packets by default),
// and fills at the rate:
//
//	1.25 * congestion_window / smoothed_rtt
//
// A sender can send one congestion window of packets per RTT,
// since the congestion window consumed by each packet is returned
// one round-trip later by the responding ack.
// The pacer permits sending at slightly faster than this rate to
// avoid underutilizing the congestion window.
//
// The pacer permits the bucket to become negative, and permits
// sending when non-negative. This biases slightly in favor of
// sending packets over limiting them, and permits bursts one
// packet greater than the configured maximum, but permits the pacer
// to be ignorant of the maximum packet size.
//
// https://www.rfc-editor.org/rfc/rfc9002.html#section-7.7
type pacerState struct {
	bucket           int // measured in bytes
	maxBucket        int
	timerGranularity time.Duration
	lastUpdate       time.Time
	nextSend         time.Time
}

func (p *pacerState) init(now time.Time, maxBurst int, timerGranularity time.Duration) {
	// Bucket is limited to maximum burst size, which is the initial congestion window.
	// https://www.rfc-editor.org/rfc/rfc9002#section-7.7-2
	p.maxBucket = maxBurst
	p.bucket = p.maxBucket
	p.timerGranularity = timerGranularity
	p.lastUpdate = now
	p.nextSend = now
}

// pacerBytesForInterval returns the number of bytes permitted over an interval.
//
//	rate  = 1.25 * congestion_window / smoothed_rtt
//	bytes = interval * rate
//
// https://www.rfc-editor.org/rfc/rfc9002#section-7.7-6
func pacerBytesForInterval(interval time.Duration, congestionWindow int, rtt time.Duration) int {
	bytes := (int64(interval) * int64(congestionWindow)) / int64(rtt)
	bytes = (bytes * 5) / 4 // bytes *= 1.25
	return int(bytes)
}

// pacerIntervalForBytes returns the amount of time required for a number of bytes.
//
//	time_per_byte = (smoothed_rtt / congestion_window) / 1.25
//	interval      = time_per_byte * bytes
//
// https://www.rfc-editor.org/rfc/rfc9002#section-7.7-8
func pacerIntervalForBytes(bytes int, congestionWindow int, rtt time.Duration) time.Duration {
	interval := (int64(rtt) * int64(bytes)) / int64(congestionWindow)
	interval = (interval * 4) / 5 // interval /= 1.25
	return time.Duration(interval)
}

// advance is called when time passes.
func (p *pacerState) advance(now time.Time, congestionWindow int, rtt time.Duration) {
	elapsed := now.Sub(p.lastUpdate)
	if elapsed < 0 {
		// Time has gone backward?
		elapsed = 0
		p.nextSend = now // allow a packet through to get back on track
		if p.bucket < 0 {
			p.bucket = 0
		}
	}
	p.lastUpdate = now
	if rtt == 0 {
		// Avoid divide by zero in the implausible case that we measure no RTT.
		p.bucket = p.maxBucket
		return
	}
	// Refill the bucket.
	delta := pacerBytesForInterval(elapsed, congestionWindow, rtt)
	p.bucket = min(p.bucket+delta, p.maxBucket)
}

// packetSent is called to record transmission of a packet.
func (p *pacerState) packetSent(now time.Time, size, congestionWindow int, rtt time.Duration) {
	p.bucket -= size
	if p.bucket < -congestionWindow {
		// Never allow the bucket to fall more than one congestion window in arrears.
		// We can only fall this far behind if the sender is sending unpaced packets,
		// the congestion window has been exceeded, or the RTT is less than the
		// timer granularity.
		//
		// Limiting the minimum bucket size limits the maximum pacer delay
		// to RTT/1.25.
		p.bucket = -congestionWindow
	}
	if p.bucket >= 0 {
		p.nextSend = now
		return
	}
	// Next send occurs when the bucket has refilled to 0.
	delay := pacerIntervalForBytes(-p.bucket, congestionWindow, rtt)
	p.nextSend = now.Add(delay)
}

// canSend reports whether a packet can be sent now.
// If it returns false, next is the time when the next packet can be sent.
func (p *pacerState) canSend(now time.Time) (canSend bool, next time.Time) {
	// If the next send time is within the timer granularity, send immediately.
	if p.nextSend.After(now.Add(p.timerGranularity)) {
		return false, p.nextSend
	}
	return true, time.Time{}
}

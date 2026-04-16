// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package quic

import (
	"math"
	"time"
)

// An unscaledAckDelay is an ACK Delay field value from an ACK packet,
// without the ack_delay_exponent scaling applied.
type unscaledAckDelay int64

func unscaledAckDelayFromDuration(d time.Duration, ackDelayExponent uint8) unscaledAckDelay {
	return unscaledAckDelay(d.Microseconds() >> ackDelayExponent)
}

func (d unscaledAckDelay) Duration(ackDelayExponent uint8) time.Duration {
	if int64(d) > (math.MaxInt64>>ackDelayExponent)/int64(time.Microsecond) {
		// If scaling the delay would overflow, ignore the delay.
		return 0
	}
	return time.Duration(d<<ackDelayExponent) * time.Microsecond
}

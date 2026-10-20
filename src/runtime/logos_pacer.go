// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "internal/runtime/atomic"

const (
	// logosElasticFloor is the background utilization below logosElasticRise.
	logosElasticFloor = 0.25

	// logosElasticCeiling is the background utilization from logosElasticSaturate on.
	logosElasticCeiling = 0.75

	// logosElasticRise is the live-over-goal ratio where the curve starts to rise.
	logosElasticRise = 0.80

	// logosElasticSaturate is the live-over-goal ratio where the curve reaches
	// logosElasticCeiling and mark assist is the only remaining backstop.
	logosElasticSaturate = 0.95

	// logosElasticSlack absorbs float rounding when the curve value times
	// GOMAXPROCS lands on a whole worker count.
	logosElasticSlack = 1e-9
)

// logosElasticState is the elastic background marking state of one
// controller.
//
// active and procs are written only when a cycle starts, with the world
// stopped. peak holds the bits of the largest background utilization seen in
// the cycle, in units of Ps. A non-negative float orders like its bits, so
// peak only rises, and the dedicated and fractional goals are both derived
// from one value of it.
type logosElasticState struct {
	active bool
	procs  int32
	peak   atomic.Uint64
}

// logosElasticUtilization is the PLAN background utilization for live heap
// over heap goal u: logosElasticFloor below logosElasticRise, a linear rise
// to logosElasticCeiling below logosElasticSaturate, and logosElasticCeiling
// from there on.
func logosElasticUtilization(u float64) float64 {
	switch {
	case u < logosElasticRise:
		return logosElasticFloor
	case u >= logosElasticSaturate:
		return logosElasticCeiling
	}
	span := (u - logosElasticRise) / (logosElasticSaturate - logosElasticRise)
	return logosElasticFloor + (logosElasticCeiling-logosElasticFloor)*span
}

// logosElasticRatio is live over goal; a zero goal reads as an empty heap.
func logosElasticRatio(live, goal uint64) float64 {
	if goal == 0 {
		return 0
	}
	return float64(live) / float64(goal)
}

// logosElasticDedicated is the number of whole dedicated workers in total
// Ps of background utilization.
func logosElasticDedicated(total float64) int64 {
	return int64(total + logosElasticSlack)
}

// logosElasticFractional is the fractional goal that makes up the part of
// total that dedicated workers do not cover. A fractional worker runs only on
// a P without a dedicated worker, so the goal is per such P and the part is
// shared by procs minus dedicated Ps. The curve never exceeds 0.75 of procs,
// which keeps at least one such P.
func logosElasticFractional(total float64, procs int32) float64 {
	dedicated := logosElasticDedicated(total)
	rest := total - float64(dedicated)
	if rest <= logosElasticSlack {
		return 0
	}
	return rest / float64(int64(procs)-dedicated)
}

// logosElasticStart replaces the worker split of a starting cycle when the
// collector mode is greentea. dedicated is the upstream split. It returns the
// number of dedicated workers to start, and sets the fractional goal for the
// rest. Any other mode, and stop-the-world collection, keep the upstream
// split. The caller holds worldsema with the world stopped.
func (c *gcControllerState) logosElasticStart(procs int, dedicated int64) int64 {
	e := &c.logosElastic
	e.active = logosGCMode == "greentea" && debug.gcstoptheworld == 0
	if !e.active {
		return dedicated
	}
	e.procs = int32(procs)
	u := logosElasticRatio(c.heapLive.Load(), c.heapGoal())
	total := float64(procs) * logosElasticUtilization(u)
	e.peak.Store(float64bits(total))
	dedicated = logosElasticDedicated(total)
	c.fractionalUtilizationGoal = logosElasticFractional(total, e.procs)
	if debug.gcpacertrace > 0 {
		print("pacer: elastic u=", u, " util=", total/float64(procs), " workers=", dedicated, "+", c.fractionalUtilizationGoal, "\n")
	}
	return dedicated
}

// logosElasticRevise follows the PLAN curve during the cycle. It is called
// from revise with the live heap and the heap goal before revise extends it.
func (c *gcControllerState) logosElasticRevise(live, goal uint64) {
	if c.logosElastic.active {
		c.logosElasticStep(live, goal)
	}
}

// logosElasticStep raises the cycle's peak utilization to the curve value at
// live over goal when the curve is above it, starts the dedicated workers the
// raise adds, and publishes the fractional goal of the raised peak.
// Utilization never falls inside a cycle, because running dedicated workers
// are not stopped.
//
// Every successful raise adds exactly the difference of the two dedicated
// counts it moves between to dedicatedMarkWorkersNeeded, so needed plus
// running dedicated workers always equals the count of the peak. The
// fractional goal is a plain float64 that readers load without a lock; a
// torn read on a 32-bit platform differs from one of the two stored values
// only in the low 32 mantissa bits.
func (c *gcControllerState) logosElasticStep(live, goal uint64) {
	e := &c.logosElastic
	total := float64(e.procs) * logosElasticUtilization(logosElasticRatio(live, goal))
	for {
		old := e.peak.Load()
		oldTotal := float64frombits(old)
		if oldTotal >= total {
			return
		}
		if e.peak.CompareAndSwap(old, float64bits(total)) {
			if add := logosElasticDedicated(total) - logosElasticDedicated(oldTotal); add != 0 {
				c.dedicatedMarkWorkersNeeded.Add(add)
			}
			c.logosElasticPublish()
			return
		}
	}
}

// logosElasticPublish stores the fractional goal of the current peak, and
// stores again when a concurrent raise moved the peak meanwhile.
func (c *gcControllerState) logosElasticPublish() {
	e := &c.logosElastic
	for {
		peak := e.peak.Load()
		c.fractionalUtilizationGoal = logosElasticFractional(float64frombits(peak), e.procs)
		if e.peak.Load() == peak {
			return
		}
	}
}

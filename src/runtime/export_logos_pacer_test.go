// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

// Test seam for M19-S2: elastic background marking, observed through the
// controller copy that export_test.go builds.

// LogosElasticUtilization is the PLAN background utilization curve.
func LogosElasticUtilization(u float64) float64 {
	return logosElasticUtilization(u)
}

// LogosPacerSetMode sets the collector mode under worldsema, so that no
// cycle start reads it half written, and returns the previous mode. Only
// the benchmark uses it, because it needs real collections under a mode.
func LogosPacerSetMode(mode string) (old string) {
	semacquire(&worldsema)
	old = logosGCMode
	logosGCMode = mode
	semrelease(&worldsema)
	return old
}

// LogosStartCycleAt starts a cycle under the collector mode with the live
// heap at u times the heap goal. The mode holds only while worldsema is
// held, as it is during the cycle start of a real collection, so no other
// code observes it.
func (c *GCController) LogosStartCycleAt(mode string, u float64, gomaxprocs int) {
	semacquire(&worldsema)
	old := logosGCMode
	logosGCMode = mode
	c.setLiveAt(u)
	c.startCycle(0, gomaxprocs, gcTrigger{kind: gcTriggerHeap})
	logosGCMode = old
	semrelease(&worldsema)
}

// LogosReviseAt moves the live heap to u times the heap goal and revises.
func (c *GCController) LogosReviseAt(u float64) {
	c.setLiveAt(u)
	c.revise()
}

// LogosRevise revises at the current live heap.
func (c *GCController) LogosRevise() {
	c.revise()
}

func (c *GCController) setLiveAt(u float64) {
	c.heapLive.Store(uint64(u*float64(c.heapGoal()) + 0.5))
}

// LogosLiveOverGoal reports the live heap over the heap goal.
func (c *GCController) LogosLiveOverGoal() float64 {
	return float64(c.heapLive.Load()) / float64(c.heapGoal())
}

// LogosBackgroundUtilization is the background mark utilization the
// controller delivers: the dedicated workers, started or still needed, plus
// the fractional goal on each of the other Ps, over procs.
func (c *GCController) LogosBackgroundUtilization(gomaxprocs int, started int64) float64 {
	dedicated := float64(c.dedicatedMarkWorkersNeeded.Load() + started)
	return (dedicated + c.fractionalUtilizationGoal*(float64(gomaxprocs)-dedicated)) / float64(gomaxprocs)
}

// LogosElasticStepAt runs the revise hook with the live heap at u times the
// heap goal, without storing the live heap, and reports the ratio it used.
func (c *GCController) LogosElasticStepAt(u float64) float64 {
	goal := c.heapGoal()
	live := uint64(u*float64(goal) + 0.5)
	c.logosElasticRevise(live, goal)
	return float64(live) / float64(goal)
}

// LogosDedicatedTarget is the elastic dedicated worker target of the
// running cycle.
func (c *GCController) LogosDedicatedTarget() int64 {
	return logosElasticDedicated(float64frombits(c.logosElastic.peak.Load()))
}

// LogosElasticActive reports whether the running cycle paces elastically.
func (c *GCController) LogosElasticActive() bool {
	return c.logosElastic.active
}

// LogosDedicatedNeeded is the number of dedicated workers still to start.
func (c *GCController) LogosDedicatedNeeded() int64 {
	return c.dedicatedMarkWorkersNeeded.Load()
}

// LogosFractionalGoal is the fractional utilization goal.
func (c *GCController) LogosFractionalGoal() float64 {
	return c.fractionalUtilizationGoal
}

// LogosStartDedicated takes one dedicated worker the way
// assignWaitingGCWorker does.
func (c *GCController) LogosStartDedicated() bool {
	for {
		v := c.dedicatedMarkWorkersNeeded.Load()
		if v <= 0 {
			return false
		}
		if c.dedicatedMarkWorkersNeeded.CompareAndSwap(v, v-1) {
			return true
		}
	}
}

// LogosStopDedicated stops one dedicated worker the way markWorkerStop
// does.
func (c *GCController) LogosStopDedicated() {
	c.markWorkerStop(gcMarkWorkerDedicatedMode, 0)
}

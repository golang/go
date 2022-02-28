// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"fmt"
	"internal/goexperiment"
	"math"
	"math/rand"
	. "runtime"
	"testing"
	"time"
)

func TestGcPacer(t *testing.T) {
	t.Parallel()

	const initialHeapBytes = 256 << 10
	for _, e := range []*gcExecTest{
		{
			// The most basic test case: a steady-state heap.
			// Growth to an O(MiB) heap, then constant heap size, alloc/scan rates.
			name:          "Steady",
			gcPercent:     100,
			globalsBytes:  32 << 10,
			nCores:        8,
			allocRate:     constant(33.0),
			scanRate:      constant(1024.0),
			growthRate:    constant(2.0).sum(ramp(-1.0, 12)),
			scannableFrac: constant(1.0),
			stackBytes:    constant(8192),
			length:        50,
			checker: func(t *testing.T, c []gcCycleResult) {
				n := len(c)
				if n >= 25 {
					if goexperiment.PacerRedesign {
						// For the pacer redesign, assert something even stronger: at this alloc/scan rate,
						// it should be extremely close to the goal utilization.
						assertInEpsilon(t, "GC utilization", c[n-1].gcUtilization, GCGoalUtilization, 0.005)
					}

					// Make sure the pacer settles into a non-degenerate state in at least 25 GC cycles.
					assertInEpsilon(t, "GC utilization", c[n-1].gcUtilization, c[n-2].gcUtilization, 0.005)
					assertInRange(t, "goal ratio", c[n-1].goalRatio(), 0.95, 1.05)
				}
			},
		},
		{
			// Same as the steady-state case, but lots of stacks to scan relative to the heap size.
			name:          "SteadyBigStacks",
			gcPercent:     100,
			globalsBytes:  32 << 10,
			nCores:        8,
			allocRate:     constant(132.0),
			scanRate:      constant(1024.0),
			growthRate:    constant(2.0).sum(ramp(-1.0, 12)),
			scannableFrac: constant(1.0),
			stackBytes:    constant(2048).sum(ramp(128<<20, 8)),
			length:        50,
			checker: func(t *testing.T, c []gcCycleResult) {
				// Check the same conditions as the steady-state case, except the old pacer can't
				// really handle this well, so don't check the goal ratio for it.
				n := len(c)
				if n >= 25 {
					if goexperiment.PacerRedesign {
						// For the pacer redesign, assert something even stronger: at this alloc/scan rate,
						// it should be extremely close to the goal utilization.
						assertInEpsilon(t, "GC utilization", c[n-1].gcUtilization, GCGoalUtilization, 0.005)
						assertInRange(t, "goal ratio", c[n-1].goalRatio(), 0.95, 1.05)
					}

					// Make sure the pacer settles into a non-degenerate state in at least 25 GC cycles.
					assertInEpsilon(t, "GC utilization", c[n-1].gcUtilization, c[n-2].gcUtilization, 0.005)
				}
			},
		},
		{
			// Same as the steady-state case, but lots of globals to scan relative to the heap size.
			name:          "SteadyBigGlobals",
			gcPercent:     100,
			globalsBytes:  128 << 20,
			nCores:        8,
			allocRate:     constant(132.0),
			scanRate:      constant(1024.0),
			growthRate:    constant(2.0).sum(ramp(-1.0, 12)),
			scannableFrac: constant(1.0),
			stackBytes:    constant(8192),
			length:        50,
			checker: func(t *testing.T, c []gcCycleResult) {
				// Check the same conditions as the steady-state case, except the old pacer can't
				// really handle this well, so don't check the goal ratio for it.
				n := len(c)
				if n >= 25 {
					if goexperiment.PacerRedesign {
						// For the pacer redesign, assert something even stronger: at this alloc/scan rate,
						// it should be extremely close to the goal utilization.
						assertInEpsilon(t, "GC utilization", c[n-1].gcUtilization, GCGoalUtilization, 0.005)
						assertInRange(t, "goal ratio", c[n-1].goalRatio(), 0.95, 1.05)
					}

					// Make sure the pacer settles into a non-degenerate state in at least 25 GC cycles.
					assertInEpsilon(t, "GC utilization", c[n-1].gcUtilization, c[n-2].gcUtilization, 0.005)
				}
			},
		},
		{
			// This tests the GC pacer's response to a small change in allocation rate.
			name:          "StepAlloc",
			gcPercent:     100,
			globalsBytes:  32 << 10,
			nCores:        8,
			allocRate:     constant(33.0).sum(ramp(66.0, 1).delay(50)),
			scanRate:      constant(1024.0),
			growthRate:    constant(2.0).sum(ramp(-1.0, 12)),
			scannableFrac: constant(1.0),
			stackBytes:    constant(8192),
			length:        100,
			checker: func(t *testing.T, c []gcCycleResult) {
				n := len(c)
				if (n >= 25 && n < 50) || n >= 75 {
					// Make sure the pacer settles into a non-degenerate state in at least 25 GC cycles
					// and then is able to settle again after a significant jump in allocation rate.
					assertInEpsilon(t, "GC utilization", c[n-1].gcUtilization, c[n-2].gcUtilization, 0.005)
					assertInRange(t, "goal ratio", c[n-1].goalRatio(), 0.95, 1.05)
				}
			},
		},
		{
			// This tests the GC pacer's response to a large change in allocation rate.
			name:          "HeavyStepAlloc",
			gcPercent:     100,
			globalsBytes:  32 << 10,
			nCores:        8,
			allocRate:     constant(33).sum(ramp(330, 1).delay(50)),
			scanRate:      constant(1024.0),
			growthRate:    constant(2.0).sum(ramp(-1.0, 12)),
			scannableFrac: constant(1.0),
			stackBytes:    constant(8192),
			length:        100,
			checker: func(t *testing.T, c []gcCycleResult) {
				n := len(c)
				if (n >= 25 && n < 50) || n >= 75 {
					// Make sure the pacer settles into a non-degenerate state in at least 25 GC cycles
					// and then is able to settle again after a significant jump in allocation rate.
					assertInEpsilon(t, "GC utilization", c[n-1].gcUtilization, c[n-2].gcUtilization, 0.005)
					assertInRange(t, "goal ratio", c[n-1].goalRatio(), 0.95, 1.05)
				}
			},
		},
		{
			// This tests the GC pacer's response to a change in the fraction of the scannable heap.
			name:          "StepScannableFrac",
			gcPercent:     100,
			globalsBytes:  32 << 10,
			nCores:        8,
			allocRate:     constant(128.0),
			scanRate:      constant(1024.0),
			growthRate:    constant(2.0).sum(ramp(-1.0, 12)),
			scannableFrac: constant(0.2).sum(unit(0.5).delay(50)),
			stackBytes:    constant(8192),
			length:        100,
			checker: func(t *testing.T, c []gcCycleResult) {
				n := len(c)
				if (n >= 25 && n < 50) || n >= 75 {
					// Make sure the pacer settles into a non-degenerate state in at least 25 GC cycles
					// and then is able to settle again after a significant jump in allocation rate.
					assertInEpsilon(t, "GC utilization", c[n-1].gcUtilization, c[n-2].gcUtilization, 0.005)
					assertInRange(t, "goal ratio", c[n-1].goalRatio(), 0.95, 1.05)
				}
			},
		},
		{
			// Tests the pacer for a high GOGC value with a large heap growth happening
			// in the middle. The purpose of the large heap growth is to check if GC
			// utilization ends up sensitive
			name:          "HighGOGC",
			gcPercent:     1500,
			globalsBytes:  32 << 10,
			nCores:        8,
			allocRate:     random(7, 0x53).offset(165),
			scanRate:      constant(1024.0),
			growthRate:    constant(2.0).sum(ramp(-1.0, 12), random(0.01, 0x1), unit(14).delay(25)),
			scannableFrac: constant(1.0),
			stackBytes:    constant(8192),
			length:        50,
			checker: func(t *testing.T, c []gcCycleResult) {
				n := len(c)
				if goexperiment.PacerRedesign && n > 12 {
					if n == 26 {
						// In the 26th cycle there's a heap growth. Overshoot is expected to maintain
						// a stable utilization, but we should *never* overshoot more than GOGC of
						// the next cycle.
						assertInRange(t, "goal ratio", c[n-1].goalRatio(), 0.90, 15)
					} else {
						// Give a wider goal range here. With such a high GOGC value we're going to be
						// forced to undershoot.
						//
						// TODO(mknyszek): Instead of placing a 0.95 limit on the trigger, make the limit
						// based on absolute bytes, that's based somewhat in how the minimum heap size
						// is determined.
						assertInRange(t, "goal ratio", c[n-1].goalRatio(), 0.90, 1.05)
					}

					// Ensure utilization remains stable despite a growth in live heap size
					// at GC #25. This test fails prior to the GC pacer redesign.
					//
					// Because GOGC is so large, we should also be really close to the goal utilization.
					assertInEpsilon(t, "GC utilization", c[n-1].gcUtilization, GCGoalUtilization, GCGoalUtilization+0.03)
					assertInEpsilon(t, "GC utilization", c[n-1].gcUtilization, c[n-2].gcUtilization, 0.03)
				}
			},
		},
		{
			// This test makes sure that in the face of a varying (in this case, oscillating) allocation
			// rate, the pacer does a reasonably good job of staying abreast of the changes.
			name:          "OscAlloc",
			gcPercent:     100,
			globalsBytes:  32 << 10,
			nCores:        8,
			allocRate:     oscillate(13, 0, 8).offset(67),
			scanRate:      constant(1024.0),
			growthRate:    constant(2.0).sum(ramp(-1.0, 12)),
			scannableFrac: constant(1.0),
			stackBytes:    constant(8192),
			length:        50,
			checker: func(t *testing.T, c []gcCycleResult) {
				n := len(c)
				if n > 12 {
					// After the 12th GC, the heap will stop growing. Now, just make sure that:
					// 1. Utilization isn't varying _too_ much, and
					// 2. The pacer is mostly keeping up with the goal.
					assertInRange(t, "goal ratio", c[n-1].goalRatio(), 0.95, 1.05)
					if goexperiment.PacerRedesign {
						assertInRange(t, "GC utilization", c[n-1].gcUtilization, 0.25, 0.3)
					} else {
						// The old pacer is messier here, and needs a lot more tolerance.
						assertInRange(t, "GC utilization", c[n-1].gcUtilization, 0.25, 0.4)
					}
				}
			},
		},
		{
			// This test is the same as OscAlloc, but instead of oscillating, the allocation rate is jittery.
			name:          "JitterAlloc",
			gcPercent:     100,
			globalsBytes:  32 << 10,
			nCores:        8,
			allocRate:     random(13, 0xf).offset(132),
			scanRate:      constant(1024.0),
			growthRate:    constant(2.0).sum(ramp(-1.0, 12), random(0.01, 0xe)),
			scannableFrac: constant(1.0),
			stackBytes:    constant(8192),
			length:        50,
			checker: func(t *testing.T, c []gcCycleResult) {
				n := len(c)
				if n > 12 {
					// After the 12th GC, the heap will stop growing. Now, just make sure that:
					// 1. Utilization isn't varying _too_ much, and
					// 2. The pacer is mostly keeping up with the goal.
					assertInRange(t, "goal ratio", c[n-1].goalRatio(), 0.95, 1.05)
					if goexperiment.PacerRedesign {
						assertInRange(t, "GC utilization", c[n-1].gcUtilization, 0.25, 0.3)
					} else {
						// The old pacer is messier here, and needs a lot more tolerance.
						assertInRange(t, "GC utilization", c[n-1].gcUtilization, 0.25, 0.4)
					}
				}
			},
		},
		{
			// This test is the same as JitterAlloc, but with a much higher allocation rate.
			// The jitter is proportionally the same.
			name:          "HeavyJitterAlloc",
			gcPercent:     100,
			globalsBytes:  32 << 10,
			nCores:        8,
			allocRate:     random(33.0, 0x0).offset(330),
			scanRate:      constant(1024.0),
			growthRate:    constant(2.0).sum(ramp(-1.0, 12), random(0.01, 0x152)),
			scannableFrac: constant(1.0),
			stackBytes:    constant(8192),
			length:        50,
			checker: func(t *testing.T, c []gcCycleResult) {
				n := len(c)
				if n > 13 {
					// After the 12th GC, the heap will stop growing. Now, just make sure that:
					// 1. Utilization isn't varying _too_ much, and
					// 2. The pacer is mostly keeping up with the goal.
					// We start at the 13th here because we want to use the 12th as a reference.
					assertInRange(t, "goal ratio", c[n-1].goalRatio(), 0.95, 1.05)
					// Unlike the other tests, GC utilization here will vary more and tend higher.
					// Just make sure it's not going too crazy.
					assertInEpsilon(t, "GC utilization", c[n-1].gcUtilization, c[n-2].gcUtilization, 0.05)
					if goexperiment.PacerRedesign {
						assertInEpsilon(t, "GC utilization", c[n-1].gcUtilization, c[11].gcUtilization, 0.05)
					} else {
						// The old pacer is messier here, and needs a little more tolerance.
						assertInEpsilon(t, "GC utilization", c[n-1].gcUtilization, c[11].gcUtilization, 0.07)
					}
				}
			},
		},
		// TODO(mknyszek): Write a test that exercises the pacer's hard goal.
		// This is difficult in the idealized model this testing framework places
		// the pacer in, because the calculated overshoot is directly proportional
		// to the runway for the case of the expected work.
		// However, it is still possible to trigger this case if something exceptional
		// happens between calls to revise; the framework just doesn't support this yet.
	} {
		e := e
		t.Run(e.name, func(t *testing.T) {
			t.Parallel()

			c := NewGCController(e.gcPercent)
			var bytesAllocatedBlackLast int64
			results := make([]gcCycleResult, 0, e.length)
			for i := 0; i < e.length; i++ {
				cycle := e.next()
				c.StartCycle(cycle.stackBytes, e.globalsBytes, cycle.scannableFrac, e.nCores)

				// Update pacer incrementally as we complete scan work.
				const (
					revisePeriod = 500 * time.Microsecond
					rateConv     = 1024 * float64(revisePeriod) / float64(time.Millisecond)
				)
				var nextHeapMarked int64
				if i == 0 {
					nextHeapMarked = initialHeapBytes
				} else {
					nextHeapMarked = int64(float64(int64(c.HeapMarked())-bytesAllocatedBlackLast) * cycle.growthRate)
				}
				globalsScanWorkLeft := int64(e.globalsBytes)
				stackScanWorkLeft := int64(cycle.stackBytes)
				heapScanWorkLeft := int64(float64(nextHeapMarked) * cycle.scannableFrac)
				doWork := func(work int64) (int64, int64, int64) {
					var deltas [3]int64

					// Do globals work first, then stacks, then heap.
					for i, workLeft := range []*int64{&globalsScanWorkLeft, &stackScanWorkLeft, &heapScanWorkLeft} {
						if *workLeft == 0 {
							continue
						}
						if *workLeft > work {
							deltas[i] += work
							*workLeft -= work
							work = 0
							break
						} else {
							deltas[i] += *workLeft
							work -= *workLeft
							*workLeft = 0
						}
					}
					return deltas[0], deltas[1], deltas[2]
				}
				var (
					gcDuration          int64
					assistTime          int64
					bytesAllocatedBlack int64
				)
				for heapScanWorkLeft+stackScanWorkLeft+globalsScanWorkLeft > 0 {
					// Simulate GC assist pacing.
					//
					// Note that this is an idealized view of the GC assist pacing
					// mechanism.

					// From the assist ratio and the alloc and scan rates, we can idealize what
					// the GC CPU utilization looks like.
					//
					// We start with assistRatio = (bytes of scan work) / (bytes of runway) (by definition).
					//
					// Over revisePeriod, we can also calculate how many bytes are scanned and
					// allocated, given some GC CPU utilization u:
					//
					//     bytesScanned   = scanRate  * rateConv * nCores * u
					//     bytesAllocated = allocRate * rateConv * nCores * (1 - u)
					//
					// During revisePeriod, assistRatio is kept constant, and GC assists kick in to
					// maintain it. Specifically, they act to prevent too many bytes being allocated
					// compared to how many bytes are scanned. It directly defines the ratio of
					// bytesScanned to bytesAllocated over this period, hence:
					//
					//     assistRatio = bytesScanned / bytesAllocated
					//
					// From this, we can solve for utilization, because everything else has already
					// been determined:
					//
					//     assistRatio = (scanRate * rateConv * nCores * u) / (allocRate * rateConv * nCores * (1 - u))
					//     assistRatio = (scanRate * u) / (allocRate * (1 - u))
					//     assistRatio * allocRate * (1-u) = scanRate * u
					//     assistRatio * allocRate - assistRatio * allocRate * u = scanRate * u
					//     assistRatio * allocRate = assistRatio * allocRate * u + scanRate * u
					//     assistRatio * allocRate = (assistRatio * allocRate + scanRate) * u
					//     u = (assistRatio * allocRate) / (assistRatio * allocRate + scanRate)
					//
					// Note that this may give a utilization that is _less_ than GCBackgroundUtilization,
					// which isn't possible in practice because of dedicated workers. Thus, this case
					// must be interpreted as GC assists not kicking in at all, and just round up. All
					// downstream values will then have this accounted for.
					assistRatio := c.AssistWorkPerByte()
					utilization := assistRatio * cycle.allocRate / (assistRatio*cycle.allocRate + cycle.scanRate)
					if utilization < GCBackgroundUtilization {
						utilization = GCBackgroundUtilization
					}

					// Knowing the utilization, calculate bytesScanned and bytesAllocated.
					bytesScanned := int64(cycle.scanRate * rateConv * float64(e.nCores) * utilization)
					bytesAllocated := int64(cycle.allocRate * rateConv * float64(e.nCores) * (1 - utilization))

					// Subtract work from our model.
					globalsScanned, stackScanned, heapScanned := doWork(bytesScanned)

					// doWork may not use all of bytesScanned.
					// In this case, the GC actually ends sometime in this period.
					// Let's figure out when, exactly, and adjust bytesAllocated too.
					actualElapsed := revisePeriod
					actualAllocated := bytesAllocated
					if actualScanned := globalsScanned + stackScanned + heapScanned; actualScanned < bytesScanned {
						// actualScanned = scanRate * rateConv * (t / revisePeriod) * nCores * u
						// => t = actualScanned * revisePeriod / (scanRate * rateConv * nCores * u)
						actualElapsed = time.Duration(float64(actualScanned) * float64(revisePeriod) / (cycle.scanRate * rateConv * float64(e.nCores) * utilization))
						actualAllocated = int64(cycle.allocRate * rateConv * float64(actualElapsed) / float64(revisePeriod) * float64(e.nCores) * (1 - utilization))
					}

					// Ask the pacer to revise.
					c.Revise(GCControllerReviseDelta{
						HeapLive:        actualAllocated,
						HeapScan:        int64(float64(actualAllocated) * cycle.scannableFrac),
						HeapScanWork:    heapScanned,
						StackScanWork:   stackScanned,
						GlobalsScanWork: globalsScanned,
					})

					// Accumulate variables.
					assistTime += int64(float64(actualElapsed) * float64(e.nCores) * (utilization - GCBackgroundUtilization))
					gcDuration += int64(actualElapsed)
					bytesAllocatedBlack += actualAllocated
				}

				// Put together the results, log them, and concatenate them.
				result := gcCycleResult{
					cycle:         i + 1,
					heapLive:      c.HeapMarked(),
					heapScannable: int64(float64(int64(c.HeapMarked())-bytesAllocatedBlackLast) * cycle.scannableFrac),
					heapTrigger:   c.Trigger(),
					heapPeak:      c.HeapLive(),
					heapGoal:      c.HeapGoal(),
					gcUtilization: float64(assistTime)/(float64(gcDuration)*float64(e.nCores)) + GCBackgroundUtilization,
				}
				t.Log("GC", result.String())
				results = append(results, result)

				// Run the checker for this test.
				e.check(t, results)

				c.EndCycle(uint64(nextHeapMarked+bytesAllocatedBlack), assistTime, gcDuration, e.nCores)

				bytesAllocatedBlackLast = bytesAllocatedBlack
			}
		})
	}
}

type gcExecTest struct {
	name string

	gcPercent    int
	globalsBytes uint64
	nCores       int

	allocRate     float64Stream // > 0, KiB / cpu-ms
	scanRate      float64Stream // > 0, KiB / cpu-ms
	growthRate    float64Stream // > 0
	scannableFrac float64Stream // Clamped to [0, 1]
	stackBytes    float64Stream // Multiple of 2048.
	length        int

	checker func(*testing.T, []gcCycleResult)
}

// minRate is an arbitrary minimum for allocRate, scanRate, and growthRate.
// These values just cannot be zero.
const minRate = 0.0001

func (e *gcExecTest) next() gcCycle {
	return gcCycle{
		allocRate:     e.allocRate.min(minRate)(),
		scanRate:      e.scanRate.min(minRate)(),
		growthRate:    e.growthRate.min(minRate)(),
		scannableFrac: e.scannableFrac.limit(0, 1)(),
		stackBytes:    uint64(e.stackBytes.quantize(2048).min(0)()),
	}
}

func (e *gcExecTest) check(t *testing.T, results []gcCycleResult) {
	t.Helper()

	// Do some basic general checks first.
	n := len(results)
	switch n {
	case 0:
		t.Fatal("no results passed to check")
		return
	case 1:
		if results[0].cycle != 1 {
			t.Error("first cycle has incorrect number")
		}
	default:
		if results[n-1].cycle != results[n-2].cycle+1 {
			t.Error("cycle numbers out of order")
		}
	}
	if u := results[n-1].gcUtilization; u < 0 || u > 1 {
		t.Fatal("GC utilization not within acceptable bounds")
	}
	if s := results[n-1].heapScannable; s < 0 {
		t.Fatal("heapScannable is negative")
	}
	if e.checker == nil {
		t.Fatal("test-specific checker is missing")
	}

	// Run the test-specific checker.
	e.checker(t, results)
}

type gcCycle struct {
	allocRate     float64
	scanRate      float64
	growthRate    float64
	scannableFrac float64
	stackBytes    uint64
}

type gcCycleResult struct {
	cycle int

	// These come directly from the pacer, so uint64.
	heapLive    uint64
	heapTrigger uint64
	heapGoal    uint64
	heapPeak    uint64

	// These are produced by the simulation, so int64 and
	// float64 are more appropriate, so that we can check for
	// bad states in the simulation.
	heapScannable int64
	gcUtilization float64
}

func (r *gcCycleResult) goalRatio() float64 {
	return float64(r.heapPeak) / float64(r.heapGoal)
}

func (r *gcCycleResult) String() string {
	return fmt.Sprintf("%d %2.1f%% %d->%d->%d (goal: %d)", r.cycle, r.gcUtilization*100, r.heapLive, r.heapTrigger, r.heapPeak, r.heapGoal)
}

func assertInEpsilon(t *testing.T, name string, a, b, epsilon float64) {
	t.Helper()
	assertInRange(t, name, a, b-epsilon, b+epsilon)
}

func assertInRange(t *testing.T, name string, a, min, max float64) {
	t.Helper()
	if a < min || a > max {
		t.Errorf("%s not in range (%f, %f): %f", name, min, max, a)
	}
}

// float64Stream is a function that generates an infinite stream of
// float64 values when called repeatedly.
type float64Stream func() float64

// constant returns a stream that generates the value c.
func constant(c float64) float64Stream {
	return func() float64 {
		return c
	}
}

// unit returns a stream that generates a single peak with
// amplitude amp, followed by zeroes.
//
// In another manner of speaking, this is the Kronecker delta.
func unit(amp float64) float64Stream {
	dropped := false
	return func() float64 {
		if dropped {
			return 0
		}
		dropped = true
		return amp
	}
}

// oscillate returns a stream that oscillates sinusoidally
// with the given amplitude, phase, and period.
func oscillate(amp, phase float64, period int) float64Stream {
	var cycle int
	return func() float64 {
		p := float64(cycle)/float64(period)*2*math.Pi + phase
		cycle++
		if cycle == period {
			cycle = 0
		}
		return math.Sin(p) * amp
	}
}

// ramp returns a stream that moves from zero to height
// over the course of length steps.
func ramp(height float64, length int) float64Stream {
	var cycle int
	return func() float64 {
		h := height * float64(cycle) / float64(length)
		if cycle < length {
			cycle++
		}
		return h
	}
}

// random returns a stream that generates random numbers
// between -amp and amp.
func random(amp float64, seed int64) float64Stream {
	r := rand.New(rand.NewSource(seed))
	return func() float64 {
		return ((r.Float64() - 0.5) * 2) * amp
	}
}

// delay returns a new stream which is a buffered version
// of f: it returns zero for cycles steps, followed by f.
func (f float64Stream) delay(cycles int) float64Stream {
	zeroes := 0
	return func() float64 {
		if zeroes < cycles {
			zeroes++
			return 0
		}
		return f()
	}
}

// scale returns a new stream that is f, but attenuated by a
// constant factor.
func (f float64Stream) scale(amt float64) float64Stream {
	return func() float64 {
		return f() * amt
	}
}

// offset returns a new stream that is f but offset by amt
// at each step.
func (f float64Stream) offset(amt float64) float64Stream {
	return func() float64 {
		old := f()
		return old + amt
	}
}

// sum returns a new stream that is the sum of all input streams
// at each step.
func (f float64Stream) sum(fs ...float64Stream) float64Stream {
	return func() float64 {
		sum := f()
		for _, s := range fs {
			sum += s()
		}
		return sum
	}
}

// quantize returns a new stream that rounds f to a multiple
// of mult at each step.
func (f float64Stream) quantize(mult float64) float64Stream {
	return func() float64 {
		r := f() / mult
		if r < 0 {
			return math.Ceil(r) * mult
		}
		return math.Floor(r) * mult
	}
}

// min returns a new stream that replaces all values produced
// by f lower than min with min.
func (f float64Stream) min(min float64) float64Stream {
	return func() float64 {
		return math.Max(min, f())
	}
}

// max returns a new stream that replaces all values produced
// by f higher than max with max.
func (f float64Stream) max(max float64) float64Stream {
	return func() float64 {
		return math.Min(max, f())
	}
}

// limit returns a new stream that replaces all values produced
// by f lower than min with min and higher than max with max.
func (f float64Stream) limit(min, max float64) float64Stream {
	return func() float64 {
		v := f()
		if v < min {
			v = min
		} else if v > max {
			v = max
		}
		return v
	}
}

func FuzzPIController(f *testing.F) {
	isNormal := func(x float64) bool {
		return !math.IsInf(x, 0) && !math.IsNaN(x)
	}
	isPositive := func(x float64) bool {
		return isNormal(x) && x > 0
	}
	// Seed with constants from controllers in the runtime.
	// It's not critical that we keep these in sync, they're just
	// reasonable seed inputs.
	f.Add(0.3375, 3.2e6, 1e9, 0.001, 1000.0, 0.01)
	f.Add(0.9, 4.0, 1000.0, -1000.0, 1000.0, 0.84)
	f.Fuzz(func(t *testing.T, kp, ti, tt, min, max, setPoint float64) {
		// Ignore uninteresting invalid parameters. These parameters
		// are constant, so in practice surprising values will be documented
		// or will be other otherwise immediately visible.
		//
		// We just want to make sure that given a non-Inf, non-NaN input,
		// we always get a non-Inf, non-NaN output.
		if !isPositive(kp) || !isPositive(ti) || !isPositive(tt) {
			return
		}
		if !isNormal(min) || !isNormal(max) || min > max {
			return
		}
		// Use a random source, but make it deterministic.
		rs := rand.New(rand.NewSource(800))
		randFloat64 := func() float64 {
			return math.Float64frombits(rs.Uint64())
		}
		p := NewPIController(kp, ti, tt, min, max)
		state := float64(0)
		for i := 0; i < 100; i++ {
			input := randFloat64()
			// Ignore the "ok" parameter. We're just trying to break it.
			// state is intentionally completely uncorrelated with the input.
			var ok bool
			state, ok = p.Next(input, setPoint, 1.0)
			if !isNormal(state) {
				t.Fatalf("got NaN or Inf result from controller: %f %v", state, ok)
			}
		}
	})
}

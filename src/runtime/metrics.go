// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

// Metrics implementation exported to runtime/metrics.

import (
	"internal/godebugs"
	"internal/runtime/atomic"
	"internal/runtime/gc"
	"unsafe"
)

var (
	// metrics is a map of runtime/metrics keys to data used by the runtime
	// to sample each metric's value. metricsInit indicates it has been
	// initialized.
	//
	// These fields are protected by metricsSema which should be
	// locked/unlocked with metricsLock() / metricsUnlock().
	metricsSema uint32 = 1
	metricsInit bool
	metrics     map[string]metricData

	sizeClassBuckets []float64
	timeHistBuckets  []float64
)

type metricData struct {
	// deps is the set of runtime statistics that this metric
	// depends on. Before compute is called, the statAggregate
	// which will be passed must ensure() these dependencies.
	deps statDepSet

	// compute is a function that populates a metricValue
	// given a populated statAggregate structure.
	compute func(in *statAggregate, out *metricValue)
}

func metricsLock() {
	// Acquire the metricsSema but with handoff. Operations are typically
	// expensive enough that queueing up goroutines and handing off between
	// them will be noticeably better-behaved.
	semacquire1(&metricsSema, true, 0, 0, waitReasonSemacquire)
	if raceenabled {
		raceacquire(unsafe.Pointer(&metricsSema))
	}
}

func metricsUnlock() {
	if raceenabled {
		racerelease(unsafe.Pointer(&metricsSema))
	}
	semrelease(&metricsSema)
}

// initMetrics initializes the metrics map if it hasn't been yet.
//
// metricsSema must be held.
func initMetrics() {
	if metricsInit {
		return
	}

	sizeClassBuckets = make([]float64, gc.NumSizeClasses, gc.NumSizeClasses+1)
	// Skip size class 0 which is a stand-in for large objects, but large
	// objects are tracked separately (and they actually get placed in
	// the last bucket, not the first).
	sizeClassBuckets[0] = 1 // The smallest allocation is 1 byte in size.
	for i := 1; i < gc.NumSizeClasses; i++ {
		// Size classes have an inclusive upper-bound
		// and exclusive lower bound (e.g. 48-byte size class is
		// (32, 48]) whereas we want and inclusive lower-bound
		// and exclusive upper-bound (e.g. 48-byte size class is
		// [33, 49)). We can achieve this by shifting all bucket
		// boundaries up by 1.
		//
		// Also, a float64 can precisely represent integers with
		// value up to 2^53 and size classes are relatively small
		// (nowhere near 2^48 even) so this will give us exact
		// boundaries.
		sizeClassBuckets[i] = float64(gc.SizeClassToSize[i] + 1)
	}
	sizeClassBuckets = append(sizeClassBuckets, float64Inf())

	timeHistBuckets = timeHistogramMetricsBuckets()
	metrics = map[string]metricData{
		"/cgo/go-to-c-calls:calls": {
			compute: func(_ *statAggregate, out *metricValue) {
				out.kind = metricKindUint64
				out.scalar = uint64(NumCgoCall())
			},
		},
		"/cpu/classes/gc/mark/assist:cpu-seconds": {
			deps: makeStatDepSet(cpuStatsDep),
			compute: func(in *statAggregate, out *metricValue) {
				out.kind = metricKindFloat64
				out.scalar = float64bits(nsToSec(in.cpuStats.GCAssistTime))
			},
		},
		"/cpu/classes/gc/mark/dedicated:cpu-seconds": {
			deps: makeStatDepSet(cpuStatsDep),
			compute: func(in *statAggregate, out *metricValue) {
				out.kind = metricKindFloat64
				out.scalar = float64bits(nsToSec(in.cpuStats.GCDedicatedTime))
			},
		},
		"/cpu/classes/gc/mark/idle:cpu-seconds": {
			deps: makeStatDepSet(cpuStatsDep),
			compute: func(in *statAggregate, out *metricValue) {
				out.kind = metricKindFloat64
				out.scalar = float64bits(nsToSec(in.cpuStats.GCIdleTime))
			},
		},
		"/cpu/classes/gc/pause:cpu-seconds": {
			deps: makeStatDepSet(cpuStatsDep),
			compute: func(in *statAggregate, out *metricValue) {
				out.kind = metricKindFloat64
				out.scalar = float64bits(nsToSec(in.cpuStats.GCPauseTime))
			},
		},
		"/cpu/classes/gc/total:cpu-seconds": {
			deps: makeStatDepSet(cpuStatsDep),
			compute: func(in *statAggregate, out *metricValue) {
				out.kind = metricKindFloat64
				out.scalar = float64bits(nsToSec(in.cpuStats.GCTotalTime))
			},
		},
		"/cpu/classes/idle:cpu-seconds": {
			deps: makeStatDepSet(cpuStatsDep),
			compute: func(in *statAggregate, out *metricValue) {
				out.kind = metricKindFloat64
				out.scalar = float64bits(nsToSec(in.cpuStats.IdleTime))
			},
		},
		"/cpu/classes/scavenge/assist:cpu-seconds": {
			deps: makeStatDepSet(cpuStatsDep),
			compute: func(in *statAggregate, out *metricValue) {
				out.kind = metricKindFloat64
				out.scalar = float64bits(nsToSec(in.cpuStats.ScavengeAssistTime))
			},
		},
		"/cpu/classes/scavenge/background:cpu-seconds": {
			deps: makeStatDepSet(cpuStatsDep),
			compute: func(in *statAggregate, out *metricValue) {
				out.kind = metricKindFloat64
				out.scalar = float64bits(nsToSec(in.cpuStats.ScavengeBgTime))
			},
		},
		"/cpu/classes/scavenge/total:cpu-seconds": {
			deps: makeStatDepSet(cpuStatsDep),
			compute: func(in *statAggregate, out *metricValue) {
				out.kind = metricKindFloat64
				out.scalar = float64bits(nsToSec(in.cpuStats.ScavengeTotalTime))
			},
		},
		"/cpu/classes/total:cpu-seconds": {
			deps: makeStatDepSet(cpuStatsDep),
			compute: func(in *statAggregate, out *metricValue) {
				out.kind = metricKindFloat64
				out.scalar = float64bits(nsToSec(in.cpuStats.TotalTime))
			},
		},
		"/cpu/classes/user:cpu-seconds": {
			deps: makeStatDepSet(cpuStatsDep),
			compute: func(in *statAggregate, out *metricValue) {
				out.kind = metricKindFloat64
				out.scalar = float64bits(nsToSec(in.cpuStats.UserTime))
			},
		},
		"/gc/cleanups/executed:cleanups": {
			deps: makeStatDepSet(finalStatsDep),
			compute: func(in *statAggregate, out *metricValue) {
				out.kind = metricKindUint64
				out.scalar = in.finalStats.cleanupsExecuted
			},
		},
		"/gc/cleanups/queued:cleanups": {
			deps: makeStatDepSet(finalStatsDep),
			compute: func(in *statAggregate, out *metricValue) {
				out.kind = metricKindUint64
				out.scalar = in.finalStats.cleanupsQueued
			},
		},
		"/gc/cycles/automatic:gc-cycles": {
			deps: makeStatDepSet(sysStatsDep),
			compute: func(in *statAggregate, out *metricValue) {
				out.kind = metricKindUint64
				out.scalar = in.sysStats.gcCyclesDone - in.sysStats.gcCyclesForced
			},
		},
		"/gc/cycles/forced:gc-cycles": {
			deps: makeStatDepSet(sysStatsDep),
			compute: func(in *statAggregate, out *metricValue) {
				out.kind = metricKindUint64
				out.scalar = in.sysStats.gcCyclesForced
			},
		},
		"/gc/cycles/total:gc-cycles": {
			deps: makeStatDepSet(sysStatsDep),
			compute: func(in *statAggregate, out *metricValue) {
				out.kind = metricKindUint64
				out.scalar = in.sysStats.gcCyclesDone
			},
		},
		"/gc/finalizers/executed:finalizers": {
			deps: makeStatDepSet(finalStatsDep),
			compute: func(in *statAggregate, out *metricValue) {
				out.kind = metricKindUint64
				out.scalar = in.finalStats.finalizersExecuted
			},
		},
		"/gc/finalizers/queued:finalizers": {
			deps: makeStatDepSet(finalStatsDep),
			compute: func(in *statAggregate, out *metricValue) {
				out.kind = metricKindUint64
				out.scalar = in.finalStats.finalizersQueued
			},
		},
		"/gc/scan/globals:bytes": {
			deps: makeStatDepSet(gcStatsDep),
			compute: func(in *statAggregate, out *metricValue) {
				out.kind = metricKindUint64
				out.scalar = in.gcStats.globalsScan
			},
		},
		"/gc/scan/heap:bytes": {
			deps: makeStatDepSet(gcStatsDep),
			compute: func(in *statAggregate, out *metricValue) {
				out.kind = metricKindUint64
				out.scalar = in.gcStats.heapScan
			},
		},
		"/gc/scan/stack:bytes": {
			deps: makeStatDepSet(gcStatsDep),
			compute: func(in *statAggregate, out *metricValue) {
				out.kind = metricKindUint64
				out.scalar = in.gcStats.stackScan
			},
		},
		"/gc/scan/total:bytes": {
			deps: makeStatDepSet(gcStatsDep),
			compute: func(in *statAggregate, out *metricValue) {
				out.kind = metricKindUint64
				out.scalar = in.gcStats.totalScan
			},
		},
		"/gc/heap/allocs-by-size:bytes": {
			deps: makeStatDepSet(heapStatsDep),
			compute: func(in *statAggregate, out *metricValue) {
				hist := out.float64HistOrInit(sizeClassBuckets)
				hist.counts[len(hist.counts)-1] = in.heapStats.largeAllocCount
				// Cut off the first index which is ostensibly for size class 0,
				// but large objects are tracked separately so it's actually unused.
				for i, count := range in.heapStats.smallAllocCount[1:] {
					hist.counts[i] = count
				}
			},
		},
		"/gc/heap/allocs:bytes": {
			deps: makeStatDepSet(heapStatsDep),
			compute: func(in *statAggregate, out *metricValue) {
				out.kind = metricKindUint64
				out.scalar = in.heapStats.totalAllocated
			},
		},
		"/gc/heap/allocs:objects": {
			deps: makeStatDepSet(heapStatsDep),
			compute: func(in *statAggregate, out *metricValue) {
				out.kind = metricKindUint64
				out.scalar = in.heapStats.totalAllocs
			},
		},
		"/gc/heap/frees-by-size:bytes": {
			deps: makeStatDepSet(heapStatsDep),
			compute: func(in *statAggregate, out *metricValue) {
				hist := out.float64HistOrInit(sizeClassBuckets)
				hist.counts[len(hist.counts)-1] = in.heapStats.largeFreeCount
				// Cut off the first index which is ostensibly for size class 0,
				// but large objects are tracked separately so it's actually unused.
				for i, count := range in.heapStats.smallFreeCount[1:] {
					hist.counts[i] = count
				}
			},
		},
		"/gc/heap/frees:bytes": {
			deps: makeStatDepSet(heapStatsDep),
			compute: func(in *statAggregate, out *metricValue) {
				out.kind = metricKindUint64
				out.scalar = in.heapStats.totalFreed
			},
		},
		"/gc/heap/frees:objects": {
			deps: makeStatDepSet(heapStatsDep),
			compute: func(in *statAggregate, out *metricValue) {
				out.kind = metricKindUint64
				out.scalar = in.heapStats.totalFrees
			},
		},
		"/gc/heap/goal:bytes": {
			deps: makeStatDepSet(sysStatsDep),
			compute: func(in *statAggregate, out *metricValue) {
				out.kind = metricKindUint64
				out.scalar = in.sysStats.heapGoal
			},
		},
		"/gc/gomemlimit:bytes": {
			compute: func(in *statAggregate, out *metricValue) {
				out.kind = metricKindUint64
				out.scalar = uint64(gcController.memoryLimit.Load())
			},
		},
		"/gc/gogc:percent": {
			compute: func(in *statAggregate, out *metricValue) {
				out.kind = metricKindUint64
				out.scalar = uint64(gcController.gcPercent.Load())
			},
		},
		"/gc/heap/live:bytes": {
			deps: makeStatDepSet(heapStatsDep),
			compute: func(in *statAggregate, out *metricValue) {
				out.kind = metricKindUint64
				out.scalar = gcController.heapMarked
			},
		},
		"/gc/heap/objects:objects": {
			deps: makeStatDepSet(heapStatsDep),
			compute: func(in *statAggregate, out *metricValue) {
				out.kind = metricKindUint64
				out.scalar = in.heapStats.numObjects
			},
		},
		"/gc/heap/tiny/allocs:objects": {
			deps: makeStatDepSet(heapStatsDep),
			compute: func(in *statAggregate, out *metricValue) {
				out.kind = metricKindUint64
				out.scalar = in.heapStats.tinyAllocCount
			},
		},
		"/gc/limiter/last-enabled:gc-cycle": {
			compute: func(_ *statAggregate, out *metricValue) {
				out.kind = metricKindUint64
				out.scalar = uint64(gcCPULimiter.lastEnabledCycle.Load())
			},
		},
		"/gc/pauses:seconds": {
			compute: func(_ *statAggregate, out *metricValue) {
				// N.B. this is identical to /sched/pauses/total/gc:seconds.
				sched.stwTotalTimeGC.write(out)
			},
		},
		"/gc/stack/starting-size:bytes": {
			compute: func(in *statAggregate, out *metricValue) {
				out.kind = metricKindUint64
				out.scalar = uint64(startingStackSize)
			},
		},
		"/memory/classes/heap/free:bytes": {
			deps: makeStatDepSet(heapStatsDep),
			compute: func(in *statAggregate, out *metricValue) {
				out.kind = metricKindUint64
				out.scalar = uint64(in.heapStats.committed - in.heapStats.inHeap -
					in.heapStats.inStacks - in.heapStats.inWorkBufs)
			},
		},
		"/memory/classes/heap/objects:bytes": {
			deps: makeStatDepSet(heapStatsDep),
			compute: func(in *statAggregate, out *metricValue) {
				out.kind = metricKindUint64
				out.scalar = in.heapStats.inObjects
			},
		},
		"/memory/classes/heap/released:bytes": {
			deps: makeStatDepSet(heapStatsDep),
			compute: func(in *statAggregate, out *metricValue) {
				out.kind = metricKindUint64
				out.scalar = uint64(in.heapStats.released)
			},
		},
		"/memory/classes/heap/stacks:bytes": {
			deps: makeStatDepSet(heapStatsDep),
			compute: func(in *statAggregate, out *metricValue) {
				out.kind = metricKindUint64
				out.scalar = uint64(in.heapStats.inStacks)
			},
		},
		"/memory/classes/heap/unused:bytes": {
			deps: makeStatDepSet(heapStatsDep),
			compute: func(in *statAggregate, out *metricValue) {
				out.kind = metricKindUint64
				out.scalar = uint64(in.heapStats.inHeap) - in.heapStats.inObjects
			},
		},
		"/memory/classes/metadata/mcache/free:bytes": {
			deps: makeStatDepSet(sysStatsDep),
			compute: func(in *statAggregate, out *metricValue) {
				out.kind = metricKindUint64
				out.scalar = in.sysStats.mCacheSys - in.sysStats.mCacheInUse
			},
		},
		"/memory/classes/metadata/mcache/inuse:bytes": {
			deps: makeStatDepSet(sysStatsDep),
			compute: func(in *statAggregate, out *metricValue) {
				out.kind = metricKindUint64
				out.scalar = in.sysStats.mCacheInUse
			},
		},
		"/memory/classes/metadata/mspan/free:bytes": {
			deps: makeStatDepSet(sysStatsDep),
			compute: func(in *statAggregate, out *metricValue) {
				out.kind = metricKindUint64
				out.scalar = in.sysStats.mSpanSys - in.sysStats.mSpanInUse
			},
		},
		"/memory/classes/metadata/mspan/inuse:bytes": {
			deps: makeStatDepSet(sysStatsDep),
			compute: func(in *statAggregate, out *metricValue) {
				out.kind = metricKindUint64
				out.scalar = in.sysStats.mSpanInUse
			},
		},
		"/memory/classes/metadata/other:bytes": {
			deps: makeStatDepSet(heapStatsDep, sysStatsDep),
			compute: func(in *statAggregate, out *metricValue) {
				out.kind = metricKindUint64
				out.scalar = uint64(in.heapStats.inWorkBufs) + in.sysStats.gcMiscSys
			},
		},
		"/memory/classes/os-stacks:bytes": {
			deps: makeStatDepSet(sysStatsDep),
			compute: func(in *statAggregate, out *metricValue) {
				out.kind = metricKindUint64
				out.scalar = in.sysStats.stacksSys
			},
		},
		"/memory/classes/other:bytes": {
			deps: makeStatDepSet(sysStatsDep),
			compute: func(in *statAggregate, out *metricValue) {
				out.kind = metricKindUint64
				out.scalar = in.sysStats.otherSys
			},
		},
		"/memory/classes/profiling/buckets:bytes": {
			deps: makeStatDepSet(sysStatsDep),
			compute: func(in *statAggregate, out *metricValue) {
				out.kind = metricKindUint64
				out.scalar = in.sysStats.buckHashSys
			},
		},
		"/memory/classes/total:bytes": {
			deps: makeStatDepSet(heapStatsDep, sysStatsDep),
			compute: func(in *statAggregate, out *metricValue) {
				out.kind = metricKindUint64
				out.scalar = uint64(in.heapStats.committed+in.heapStats.released) +
					in.sysStats.stacksSys + in.sysStats.mSpanSys +
					in.sysStats.mCacheSys + in.sysStats.buckHashSys +
					in.sysStats.gcMiscSys + in.sysStats.otherSys
			},
		},
		"/sched/gomaxprocs:threads": {
			compute: func(_ *statAggregate, out *metricValue) {
				out.kind = metricKindUint64
				out.scalar = uint64(gomaxprocs)
			},
		},
		"/sched/goroutines:goroutines": {
			deps: makeStatDepSet(schedStatsDep),
			compute: func(in *statAggregate, out *metricValue) {
				out.kind = metricKindUint64
				out.scalar = uint64(in.schedStats.gTotal)
			},
		},
		"/sched/goroutines/not-in-go:goroutines": {
			deps: makeStatDepSet(schedStatsDep),
			compute: func(in *statAggregate, out *metricValue) {
				out.kind = metricKindUint64
				out.scalar = uint64(in.schedStats.gNonGo)
			},
		},
		"/sched/goroutines/running:goroutines": {
			deps: makeStatDepSet(schedStatsDep),
			compute: func(in *statAggregate, out *metricValue) {
				out.kind = metricKindUint64
				out.scalar = uint64(in.schedStats.gRunning)
			},
		},
		"/sched/goroutines/runnable:goroutines": {
			deps: makeStatDepSet(schedStatsDep),
			compute: func(in *statAggregate, out *metricValue) {
				out.kind = metricKindUint64
				out.scalar = uint64(in.schedStats.gRunnable)
			},
		},
		"/sched/goroutines/waiting:goroutines": {
			deps: makeStatDepSet(schedStatsDep),
			compute: func(in *statAggregate, out *metricValue) {
				out.kind = metricKindUint64
				out.scalar = uint64(in.schedStats.gWaiting)
			},
		},
		"/sched/latencies:seconds": {
			compute: func(_ *statAggregate, out *metricValue) {
				sched.timeToRun.write(out)
			},
		},
		"/sched/pauses/stopping/gc:seconds": {
			compute: func(_ *statAggregate, out *metricValue) {
				sched.stwStoppingTimeGC.write(out)
			},
		},
		"/sched/pauses/stopping/other:seconds": {
			compute: func(_ *statAggregate, out *metricValue) {
				sched.stwStoppingTimeOther.write(out)
			},
		},
		"/sched/pauses/total/gc:seconds": {
			compute: func(_ *statAggregate, out *metricValue) {
				sched.stwTotalTimeGC.write(out)
			},
		},
		"/sched/pauses/total/other:seconds": {
			compute: func(_ *statAggregate, out *metricValue) {
				sched.stwTotalTimeOther.write(out)
			},
		},
		"/sync/mutex/wait/total:seconds": {
			compute: func(_ *statAggregate, out *metricValue) {
				out.kind = metricKindFloat64
				out.scalar = float64bits(nsToSec(totalMutexWaitTimeNanos()))
			},
		},
	}

	for _, info := range godebugs.All {
		if !info.Opaque {
			metrics["/godebug/non-default-behavior/"+info.Name+":events"] = metricData{compute: compute0}
		}
	}

	metricsInit = true
}

func compute0(_ *statAggregate, out *metricValue) {
	out.kind = metricKindUint64
	out.scalar = 0
}

type metricReader func() uint64

func (f metricReader) compute(_ *statAggregate, out *metricValue) {
	out.kind = metricKindUint64
	out.scalar = f()
}

//go:linkname godebug_registerMetric internal/godebug.registerMetric
func godebug_registerMetric(name string, read func() uint64) {
	metricsLock()
	initMetrics()
	d, ok := metrics[name]
	if !ok {
		throw("runtime: unexpected metric registration for " + name)
	}
	d.compute = metricReader(read).compute
	metrics[name] = d
	metricsUnlock()
}

// statDep is a dependency on a group of statistics
// that a metric might have.
type statDep uint

const (
	heapStatsDep  statDep = iota // corresponds to heapStatsAggregate
	sysStatsDep                  // corresponds to sysStatsAggregate
	cpuStatsDep                  // corresponds to cpuStatsAggregate
	gcStatsDep                   // corresponds to gcStatsAggregate
	finalStatsDep                // corresponds to finalStatsAggregate
	schedStatsDep                // corresponds to schedStatsAggregate
	numStatsDeps
)

// statDepSet represents a set of statDeps.
//
// Under the hood, it's a bitmap.
type statDepSet [1]uint64

// makeStatDepSet creates a new statDepSet from a list of statDeps.
func makeStatDepSet(deps ...statDep) statDepSet {
	var s statDepSet
	for _, d := range deps {
		s[d/64] |= 1 << (d % 64)
	}
	return s
}

// difference returns set difference of s from b as a new set.
func (s statDepSet) difference(b statDepSet) statDepSet {
	var c statDepSet
	for i := range s {
		c[i] = s[i] &^ b[i]
	}
	return c
}

// union returns the union of the two sets as a new set.
func (s statDepSet) union(b statDepSet) statDepSet {
	var c statDepSet
	for i := range s {
		c[i] = s[i] | b[i]
	}
	return c
}

// empty returns true if there are no dependencies in the set.
func (s *statDepSet) empty() bool {
	for _, c := range s {
		if c != 0 {
			return false
		}
	}
	return true
}

// has returns true if the set contains a given statDep.
func (s *statDepSet) has(d statDep) bool {
	return s[d/64]&(1<<(d%64)) != 0
}

// heapStatsAggregate represents memory stats obtained from the
// runtime. This set of stats is grouped together because they
// depend on each other in some way to make sense of the runtime's
// current heap memory use. They're also sharded across Ps, so it
// makes sense to grab them all at once.
type heapStatsAggregate struct {
	heapStatsDelta

	// Derived from values in heapStatsDelta.

	// inObjects is the bytes of memory occupied by objects,
	inObjects uint64

	// numObjects is the number of live objects in the heap.
	numObjects uint64

	// totalAllocated is the total bytes of heap objects allocated
	// over the lifetime of the program.
	totalAllocated uint64

	// totalFreed is the total bytes of heap objects freed
	// over the lifetime of the program.
	totalFreed uint64

	// totalAllocs is the number of heap objects allocated over
	// the lifetime of the program.
	totalAllocs uint64

	// totalFrees is the number of heap objects freed over
	// the lifetime of the program.
	totalFrees uint64
}

// compute populates the heapStatsAggregate with values from the runtime.
func (a *heapStatsAggregate) compute() {
	memstats.heapStats.read(&a.heapStatsDelta)

	// Calculate derived stats.
	a.totalAllocs = a.largeAllocCount
	a.totalFrees = a.largeFreeCount
	a.totalAllocated = a.largeAlloc
	a.totalFreed = a.largeFree
	for i := range a.smallAllocCount {
		na := a.smallAllocCount[i]
		nf := a.smallFreeCount[i]
		a.totalAllocs += na
		a.totalFrees += nf
		a.totalAllocated += na * uint64(gc.SizeClassToSize[i])
		a.totalFreed += nf * uint64(gc.SizeClassToSize[i])
	}
	a.inObjects = a.totalAllocated - a.totalFreed
	a.numObjects = a.totalAllocs - a.totalFrees
}

// sysStatsAggregate represents system memory stats obtained
// from the runtime. This set of stats is grouped together because
// they're all relatively cheap to acquire and generally independent
// of one another and other runtime memory stats. The fact that they
// may be acquired at different times, especially with respect to
// heapStatsAggregate, means there could be some skew, but because of
// these stats are independent, there's no real consistency issue here.
type sysStatsAggregate struct {
	stacksSys      uint64
	mSpanSys       uint64
	mSpanInUse     uint64
	mCacheSys      uint64
	mCacheInUse    uint64
	buckHashSys    uint64
	gcMiscSys      uint64
	otherSys       uint64
	heapGoal       uint64
	gcCyclesDone   uint64
	gcCyclesForced uint64
}

// compute populates the sysStatsAggregate with values from the runtime.
func (a *sysStatsAggregate) compute() {
	a.stacksSys = memstats.stacks_sys.load()
	a.buckHashSys = memstats.buckhash_sys.load()
	a.gcMiscSys = memstats.gcMiscSys.load()
	a.otherSys = memstats.other_sys.load()
	a.heapGoal = gcController.heapGoal()
	a.gcCyclesDone = uint64(memstats.numgc)
	a.gcCyclesForced = uint64(memstats.numforcedgc)

	systemstack(func() {
		lock(&mheap_.lock)
		a.mSpanSys = memstats.mspan_sys.load()
		a.mSpanInUse = uint64(mheap_.spanalloc.inuse)
		a.mCacheSys = memstats.mcache_sys.load()
		a.mCacheInUse = uint64(mheap_.cachealloc.inuse)
		unlock(&mheap_.lock)
	})
}

// cpuStatsAggregate represents CPU stats obtained from the runtime
// acquired together to avoid skew and inconsistencies.
type cpuStatsAggregate struct {
	cpuStats
}

// compute populates the cpuStatsAggregate with values from the runtime.
func (a *cpuStatsAggregate) compute() {
	a.cpuStats = work.cpuStats
	// TODO(mknyszek): Update the CPU stats again so that we're not
	// just relying on the STW snapshot. The issue here is that currently
	// this will cause non-monotonicity in the "user" CPU time metric.
	//
	// a.cpuStats.accumulate(nanotime(), gcphase == _GCmark)
}

// gcStatsAggregate represents various GC stats obtained from the runtime
// acquired together to avoid skew and inconsistencies.
type gcStatsAggregate struct {
	heapScan    uint64
	stackScan   uint64
	globalsScan uint64
	totalScan   uint64
}

// compute populates the gcStatsAggregate with values from the runtime.
func (a *gcStatsAggregate) compute() {
	a.heapScan = gcController.heapScan.Load()
	a.stackScan = gcController.lastStackScan.Load()
	a.globalsScan = gcController.globalsScan.Load()
	a.totalScan = a.heapScan + a.stackScan + a.globalsScan
}

// finalStatsAggregate represents various finalizer/cleanup stats obtained
// from the runtime acquired together to avoid skew and inconsistencies.
type finalStatsAggregate struct {
	finalizersQueued   uint64
	finalizersExecuted uint64
	cleanupsQueued     uint64
	cleanupsExecuted   uint64
}

// compute populates the finalStatsAggregate with values from the runtime.
func (a *finalStatsAggregate) compute() {
	a.finalizersQueued, a.finalizersExecuted = finReadQueueStats()
	a.cleanupsQueued, a.cleanupsExecuted = gcCleanups.readQueueStats()
}

// schedStatsAggregate contains stats about the scheduler, including
// an approximate count of goroutines in each state.
type schedStatsAggregate struct {
	gTotal    uint64
	gRunning  uint64
	gRunnable uint64
	gNonGo    uint64
	gWaiting  uint64
}

// compute populates the schedStatsAggregate with values from the runtime.
func (a *schedStatsAggregate) compute() {
	// Lock the scheduler so the global run queue can't change and
	// the number of Ps can't change. This doesn't prevent the
	// local run queues from changing, so the results are still
	// approximate.
	lock(&sched.lock)

	// Collect running/runnable from per-P run queues.
	for _, p := range allp {
		if p == nil || p.status == _Pdead {
			break
		}
		switch p.status {
		case _Prunning:
			a.gRunning++
		case _Psyscall:
			a.gNonGo++
		case _Pgcstop:
			// The world is stopping or stopped.
			// This is fine. The results will be
			// slightly odd since nothing else
			// is running, but it will be accurate.
		}

		for {
			h := atomic.Load(&p.runqhead)
			t := atomic.Load(&p.runqtail)
			next := atomic.Loaduintptr((*uintptr)(&p.runnext))
			runnable := int32(t - h)
			if atomic.Load(&p.runqhead) != h || runnable < 0 {
				continue
			}
			if next != 0 {
				runnable++
			}
			a.gRunnable += uint64(runnable)
			break
		}
	}

	// Global run queue.
	a.gRunnable += uint64(sched.runq.size)

	// Account for Gs that are in _Gsyscall without a P in _Psyscall.
	nGsyscallNoP := sched.nGsyscallNoP.Load()

	// nGsyscallNoP can go negative during temporary races.
	if nGsyscallNoP >= 0 {
		a.gNonGo += uint64(nGsyscallNoP)
	}

	// Compute the number of blocked goroutines. We have to
	// include system goroutines in this count because we included
	// them above.
	a.gTotal = uint64(gcount(true))
	a.gWaiting = a.gTotal - (a.gRunning + a.gRunnable + a.gNonGo)
	if a.gWaiting < 0 {
		a.gWaiting = 0
	}

	unlock(&sched.lock)
}

// nsToSec takes a duration in nanoseconds and converts it to seconds as
// a float64.
func nsToSec(ns int64) float64 {
	return float64(ns) / 1e9
}

// statAggregate is the main driver of the metrics implementation.
//
// It contains multiple aggregates of runtime statistics, as well
// as a set of these aggregates that it has populated. The aggregates
// are populated lazily by its ensure method.
type statAggregate struct {
	ensured    statDepSet
	heapStats  heapStatsAggregate
	sysStats   sysStatsAggregate
	cpuStats   cpuStatsAggregate
	gcStats    gcStatsAggregate
	finalStats finalStatsAggregate
	schedStats schedStatsAggregate
}

// ensure populates statistics aggregates determined by deps if they
// haven't yet been populated.
func (a *statAggregate) ensure(deps *statDepSet) {
	missing := deps.difference(a.ensured)
	if missing.empty() {
		return
	}
	for i := statDep(0); i < numStatsDeps; i++ {
		if !missing.has(i) {
			continue
		}
		switch i {
		case heapStatsDep:
			a.heapStats.compute()
		case sysStatsDep:
			a.sysStats.compute()
		case cpuStatsDep:
			a.cpuStats.compute()
		case gcStatsDep:
			a.gcStats.compute()
		case finalStatsDep:
			a.finalStats.compute()
		case schedStatsDep:
			a.schedStats.compute()
		}
	}
	a.ensured = a.ensured.union(missing)
}

// metricKind is a runtime copy of runtime/metrics.ValueKind and
// must be kept structurally identical to that type.
type metricKind int

const (
	// These values must be kept identical to their corresponding Kind* values
	// in the runtime/metrics package.
	metricKindBad metricKind = iota
	metricKindUint64
	metricKindFloat64
	metricKindFloat64Histogram
)

// metricSample is a runtime copy of runtime/metrics.Sample and
// must be kept structurally identical to that type.
type metricSample struct {
	name  string
	value metricValue
}

// metricValue is a runtime copy of runtime/metrics.Sample and
// must be kept structurally identical to that type.
type metricValue struct {
	kind    metricKind
	scalar  uint64         // contains scalar values for scalar Kinds.
	pointer unsafe.Pointer // contains non-scalar values.
}

// float64HistOrInit tries to pull out an existing float64Histogram
// from the value, but if none exists, then it allocates one with
// the given buckets.
func (v *metricValue) float64HistOrInit(buckets []float64) *metricFloat64Histogram {
	var hist *metricFloat64Histogram
	if v.kind == metricKindFloat64Histogram && v.pointer != nil {
		hist = (*metricFloat64Histogram)(v.pointer)
	} else {
		v.kind = metricKindFloat64Histogram
		hist = new(metricFloat64Histogram)
		v.pointer = unsafe.Pointer(hist)
	}
	hist.buckets = buckets
	if len(hist.counts) != len(hist.buckets)-1 {
		hist.counts = make([]uint64, len(buckets)-1)
	}
	return hist
}

// metricFloat64Histogram is a runtime copy of runtime/metrics.Float64Histogram
// and must be kept structurally identical to that type.
type metricFloat64Histogram struct {
	counts  []uint64
	buckets []float64
}

// agg is used by readMetrics, and is protected by metricsSema.
//
// Managed as a global variable because its pointer will be
// an argument to a dynamically-defined function, and we'd
// like to avoid it escaping to the heap.
var agg statAggregate

type metricName struct {
	name string
	kind metricKind
}

// readMetricNames is the implementation of runtime/metrics.readMetricNames,
// used by the runtime/metrics test and otherwise unreferenced.
//
//go:linkname readMetricNames runtime/metrics_test.runtime_readMetricNames
func readMetricNames() []string {
	metricsLock()
	initMetrics()
	n := len(metrics)
	metricsUnlock()

	list := make([]string, 0, n)

	metricsLock()
	for name := range metrics {
		list = append(list, name)
	}
	metricsUnlock()

	return list
}

// readMetrics is the implementation of runtime/metrics.Read.
//
//go:linkname readMetrics runtime/metrics.runtime_readMetrics
func readMetrics(samplesp unsafe.Pointer, len int, cap int) {
	metricsLock()

	// Ensure the map is initialized.
	initMetrics()

	// Read the metrics.
	readMetricsLocked(samplesp, len, cap)
	metricsUnlock()
}

// readMetricsLocked is the internal, locked portion of readMetrics.
//
// Broken out for more robust testing. metricsLock must be held and
// initMetrics must have been called already.
func readMetricsLocked(samplesp unsafe.Pointer, len int, cap int) {
	// Construct a slice from the args.
	sl := slice{samplesp, len, cap}
	samples := *(*[]metricSample)(unsafe.Pointer(&sl))

	// Clear agg defensively.
	agg = statAggregate{}

	// Sample.
	for i := range samples {
		sample := &samples[i]
		data, ok := metrics[sample.name]
		if !ok {
			sample.value.kind = metricKindBad
			continue
		}
		// Ensure we have all the stats we need.
		// agg is populated lazily.
		agg.ensure(&data.deps)

		// Compute the value based on the stats we have.
		data.compute(&agg, &sample.value)
	}
}

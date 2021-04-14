// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

// Metrics implementation exported to runtime/metrics.

import (
	"runtime/internal/atomic"
	"unsafe"
)

var (
	// metrics is a map of runtime/metrics keys to
	// data used by the runtime to sample each metric's
	// value.
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

// initMetrics initializes the metrics map if it hasn't been yet.
//
// metricsSema must be held.
func initMetrics() {
	if metricsInit {
		return
	}

	sizeClassBuckets = make([]float64, _NumSizeClasses, _NumSizeClasses+1)
	// Skip size class 0 which is a stand-in for large objects, but large
	// objects are tracked separately (and they actually get placed in
	// the last bucket, not the first).
	sizeClassBuckets[0] = 1 // The smallest allocation is 1 byte in size.
	for i := 1; i < _NumSizeClasses; i++ {
		// Size classes have an inclusive upper-bound
		// and exclusive lower bound (e.g. 48-byte size class is
		// (32, 48]) whereas we want and inclusive lower-bound
		// and exclusive upper-bound (e.g. 48-byte size class is
		// [33, 49). We can achieve this by shifting all bucket
		// boundaries up by 1.
		//
		// Also, a float64 can precisely represent integers with
		// value up to 2^53 and size classes are relatively small
		// (nowhere near 2^48 even) so this will give us exact
		// boundaries.
		sizeClassBuckets[i] = float64(class_to_size[i] + 1)
	}
	sizeClassBuckets = append(sizeClassBuckets, float64Inf())

	timeHistBuckets = timeHistogramMetricsBuckets()
	metrics = map[string]metricData{
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
		"/gc/heap/allocs-by-size:bytes": {
			deps: makeStatDepSet(heapStatsDep),
			compute: func(in *statAggregate, out *metricValue) {
				hist := out.float64HistOrInit(sizeClassBuckets)
				hist.counts[len(hist.counts)-1] = uint64(in.heapStats.largeAllocCount)
				// Cut off the first index which is ostensibly for size class 0,
				// but large objects are tracked separately so it's actually unused.
				for i, count := range in.heapStats.smallAllocCount[1:] {
					hist.counts[i] = uint64(count)
				}
			},
		},
		"/gc/heap/frees-by-size:bytes": {
			deps: makeStatDepSet(heapStatsDep),
			compute: func(in *statAggregate, out *metricValue) {
				hist := out.float64HistOrInit(sizeClassBuckets)
				hist.counts[len(hist.counts)-1] = uint64(in.heapStats.largeFreeCount)
				// Cut off the first index which is ostensibly for size class 0,
				// but large objects are tracked separately so it's actually unused.
				for i, count := range in.heapStats.smallFreeCount[1:] {
					hist.counts[i] = uint64(count)
				}
			},
		},
		"/gc/heap/goal:bytes": {
			deps: makeStatDepSet(sysStatsDep),
			compute: func(in *statAggregate, out *metricValue) {
				out.kind = metricKindUint64
				out.scalar = in.sysStats.heapGoal
			},
		},
		"/gc/heap/objects:objects": {
			deps: makeStatDepSet(heapStatsDep),
			compute: func(in *statAggregate, out *metricValue) {
				out.kind = metricKindUint64
				out.scalar = in.heapStats.numObjects
			},
		},
		"/gc/pauses:seconds": {
			compute: func(_ *statAggregate, out *metricValue) {
				hist := out.float64HistOrInit(timeHistBuckets)
				// The bottom-most bucket, containing negative values, is tracked
				// as a separately as underflow, so fill that in manually and then
				// iterate over the rest.
				hist.counts[0] = atomic.Load64(&memstats.gcPauseDist.underflow)
				for i := range memstats.gcPauseDist.counts {
					hist.counts[i+1] = atomic.Load64(&memstats.gcPauseDist.counts[i])
				}
			},
		},
		"/memory/classes/heap/free:bytes": {
			deps: makeStatDepSet(heapStatsDep),
			compute: func(in *statAggregate, out *metricValue) {
				out.kind = metricKindUint64
				out.scalar = uint64(in.heapStats.committed - in.heapStats.inHeap -
					in.heapStats.inStacks - in.heapStats.inWorkBufs -
					in.heapStats.inPtrScalarBits)
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
				out.scalar = uint64(in.heapStats.inWorkBufs+in.heapStats.inPtrScalarBits) + in.sysStats.gcMiscSys
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
		"/sched/goroutines:goroutines": {
			compute: func(_ *statAggregate, out *metricValue) {
				out.kind = metricKindUint64
				out.scalar = uint64(gcount())
			},
		},
	}
	metricsInit = true
}

// statDep is a dependency on a group of statistics
// that a metric might have.
type statDep uint

const (
	heapStatsDep statDep = iota // corresponds to heapStatsAggregate
	sysStatsDep                 // corresponds to sysStatsAggregate
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

// differennce returns set difference of s from b as a new set.
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
}

// compute populates the heapStatsAggregate with values from the runtime.
func (a *heapStatsAggregate) compute() {
	memstats.heapStats.read(&a.heapStatsDelta)

	// Calculate derived stats.
	a.inObjects = uint64(a.largeAlloc - a.largeFree)
	a.numObjects = uint64(a.largeAllocCount - a.largeFreeCount)
	for i := range a.smallAllocCount {
		n := uint64(a.smallAllocCount[i] - a.smallFreeCount[i])
		a.inObjects += n * uint64(class_to_size[i])
		a.numObjects += n
	}
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
	a.heapGoal = atomic.Load64(&memstats.next_gc)
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

// statAggregate is the main driver of the metrics implementation.
//
// It contains multiple aggregates of runtime statistics, as well
// as a set of these aggregates that it has populated. The aggergates
// are populated lazily by its ensure method.
type statAggregate struct {
	ensured   statDepSet
	heapStats heapStatsAggregate
	sysStats  sysStatsAggregate
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
		}
	}
	a.ensured = a.ensured.union(missing)
}

// metricValidKind is a runtime copy of runtime/metrics.ValueKind and
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

// readMetrics is the implementation of runtime/metrics.Read.
//
//go:linkname readMetrics runtime/metrics.runtime_readMetrics
func readMetrics(samplesp unsafe.Pointer, len int, cap int) {
	// Construct a slice from the args.
	sl := slice{samplesp, len, cap}
	samples := *(*[]metricSample)(unsafe.Pointer(&sl))

	// Acquire the metricsSema but with handoff. This operation
	// is expensive enough that queueing up goroutines and handing
	// off between them will be noticably better-behaved.
	semacquire1(&metricsSema, true, 0, 0)

	// Ensure the map is initialized.
	initMetrics()

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

	semrelease(&metricsSema)
}

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"reflect"
	"runtime"
	"runtime/debug"
	"runtime/metrics"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"
	"unsafe"
)

func prepareAllMetricsSamples() (map[string]metrics.Description, []metrics.Sample) {
	all := metrics.All()
	samples := make([]metrics.Sample, len(all))
	descs := make(map[string]metrics.Description)
	for i := range all {
		samples[i].Name = all[i].Name
		descs[all[i].Name] = all[i]
	}
	return descs, samples
}

func TestReadMetrics(t *testing.T) {
	// Run a GC cycle to get some of the stats to be non-zero.
	runtime.GC()

	// Set an arbitrary memory limit to check the metric for it
	limit := int64(512 * 1024 * 1024)
	oldLimit := debug.SetMemoryLimit(limit)
	defer debug.SetMemoryLimit(oldLimit)

	// Set an GC percent to check the metric for it
	gcPercent := 99
	oldGCPercent := debug.SetGCPercent(gcPercent)
	defer debug.SetGCPercent(oldGCPercent)

	// Tests whether readMetrics produces values aligning
	// with ReadMemStats while the world is stopped.
	var mstats runtime.MemStats
	_, samples := prepareAllMetricsSamples()
	runtime.ReadMetricsSlow(&mstats, unsafe.Pointer(&samples[0]), len(samples), cap(samples))

	checkUint64 := func(t *testing.T, m string, got, want uint64) {
		t.Helper()
		if got != want {
			t.Errorf("metric %q: got %d, want %d", m, got, want)
		}
	}

	// Check to make sure the values we read line up with other values we read.
	var allocsBySize *metrics.Float64Histogram
	var tinyAllocs uint64
	var mallocs, frees uint64
	for i := range samples {
		switch name := samples[i].Name; name {
		case "/cgo/go-to-c-calls:calls":
			checkUint64(t, name, samples[i].Value.Uint64(), uint64(runtime.NumCgoCall()))
		case "/memory/classes/heap/free:bytes":
			checkUint64(t, name, samples[i].Value.Uint64(), mstats.HeapIdle-mstats.HeapReleased)
		case "/memory/classes/heap/released:bytes":
			checkUint64(t, name, samples[i].Value.Uint64(), mstats.HeapReleased)
		case "/memory/classes/heap/objects:bytes":
			checkUint64(t, name, samples[i].Value.Uint64(), mstats.HeapAlloc)
		case "/memory/classes/heap/unused:bytes":
			checkUint64(t, name, samples[i].Value.Uint64(), mstats.HeapInuse-mstats.HeapAlloc)
		case "/memory/classes/heap/stacks:bytes":
			checkUint64(t, name, samples[i].Value.Uint64(), mstats.StackInuse)
		case "/memory/classes/metadata/mcache/free:bytes":
			checkUint64(t, name, samples[i].Value.Uint64(), mstats.MCacheSys-mstats.MCacheInuse)
		case "/memory/classes/metadata/mcache/inuse:bytes":
			checkUint64(t, name, samples[i].Value.Uint64(), mstats.MCacheInuse)
		case "/memory/classes/metadata/mspan/free:bytes":
			checkUint64(t, name, samples[i].Value.Uint64(), mstats.MSpanSys-mstats.MSpanInuse)
		case "/memory/classes/metadata/mspan/inuse:bytes":
			checkUint64(t, name, samples[i].Value.Uint64(), mstats.MSpanInuse)
		case "/memory/classes/metadata/other:bytes":
			checkUint64(t, name, samples[i].Value.Uint64(), mstats.GCSys)
		case "/memory/classes/os-stacks:bytes":
			checkUint64(t, name, samples[i].Value.Uint64(), mstats.StackSys-mstats.StackInuse)
		case "/memory/classes/other:bytes":
			checkUint64(t, name, samples[i].Value.Uint64(), mstats.OtherSys)
		case "/memory/classes/profiling/buckets:bytes":
			checkUint64(t, name, samples[i].Value.Uint64(), mstats.BuckHashSys)
		case "/memory/classes/total:bytes":
			checkUint64(t, name, samples[i].Value.Uint64(), mstats.Sys)
		case "/gc/heap/allocs-by-size:bytes":
			hist := samples[i].Value.Float64Histogram()
			// Skip size class 0 in BySize, because it's always empty and not represented
			// in the histogram.
			for i, sc := range mstats.BySize[1:] {
				if b, s := hist.Buckets[i+1], float64(sc.Size+1); b != s {
					t.Errorf("bucket does not match size class: got %f, want %f", b, s)
					// The rest of the checks aren't expected to work anyway.
					continue
				}
				if c, m := hist.Counts[i], sc.Mallocs; c != m {
					t.Errorf("histogram counts do not much BySize for class %d: got %d, want %d", i, c, m)
				}
			}
			allocsBySize = hist
		case "/gc/heap/allocs:bytes":
			checkUint64(t, name, samples[i].Value.Uint64(), mstats.TotalAlloc)
		case "/gc/heap/frees-by-size:bytes":
			hist := samples[i].Value.Float64Histogram()
			// Skip size class 0 in BySize, because it's always empty and not represented
			// in the histogram.
			for i, sc := range mstats.BySize[1:] {
				if b, s := hist.Buckets[i+1], float64(sc.Size+1); b != s {
					t.Errorf("bucket does not match size class: got %f, want %f", b, s)
					// The rest of the checks aren't expected to work anyway.
					continue
				}
				if c, f := hist.Counts[i], sc.Frees; c != f {
					t.Errorf("histogram counts do not match BySize for class %d: got %d, want %d", i, c, f)
				}
			}
		case "/gc/heap/frees:bytes":
			checkUint64(t, name, samples[i].Value.Uint64(), mstats.TotalAlloc-mstats.HeapAlloc)
		case "/gc/heap/tiny/allocs:objects":
			// Currently, MemStats adds tiny alloc count to both Mallocs AND Frees.
			// The reason for this is because MemStats couldn't be extended at the time
			// but there was a desire to have Mallocs at least be a little more representative,
			// while having Mallocs - Frees still represent a live object count.
			// Unfortunately, MemStats doesn't actually export a large allocation count,
			// so it's impossible to pull this number out directly.
			//
			// Check tiny allocation count outside of this loop, by using the allocs-by-size
			// histogram in order to figure out how many large objects there are.
			tinyAllocs = samples[i].Value.Uint64()
			// Because the next two metrics tests are checking against Mallocs and Frees,
			// we can't check them directly for the same reason: we need to account for tiny
			// allocations included in Mallocs and Frees.
		case "/gc/heap/allocs:objects":
			mallocs = samples[i].Value.Uint64()
		case "/gc/heap/frees:objects":
			frees = samples[i].Value.Uint64()
		case "/gc/heap/live:bytes":
			// Check for "obviously wrong" values. We can't check a stronger invariant,
			// such as live <= HeapAlloc, because live is not 100% accurate. It's computed
			// under racy conditions, and some objects may be double-counted (this is
			// intentional and necessary for GC performance).
			//
			// Instead, check against a much more reasonable upper-bound: the amount of
			// mapped heap memory. We can't possibly overcount to the point of exceeding
			// total mapped heap memory, except if there's an accounting bug.
			if live := samples[i].Value.Uint64(); live > mstats.HeapSys {
				t.Errorf("live bytes: %d > heap sys: %d", live, mstats.HeapSys)
			} else if live == 0 {
				// Might happen if we don't call runtime.GC() above.
				t.Error("live bytes is 0")
			}
		case "/gc/gomemlimit:bytes":
			checkUint64(t, name, samples[i].Value.Uint64(), uint64(limit))
		case "/gc/heap/objects:objects":
			checkUint64(t, name, samples[i].Value.Uint64(), mstats.HeapObjects)
		case "/gc/heap/goal:bytes":
			checkUint64(t, name, samples[i].Value.Uint64(), mstats.NextGC)
		case "/gc/gogc:percent":
			checkUint64(t, name, samples[i].Value.Uint64(), uint64(gcPercent))
		case "/gc/cycles/automatic:gc-cycles":
			checkUint64(t, name, samples[i].Value.Uint64(), uint64(mstats.NumGC-mstats.NumForcedGC))
		case "/gc/cycles/forced:gc-cycles":
			checkUint64(t, name, samples[i].Value.Uint64(), uint64(mstats.NumForcedGC))
		case "/gc/cycles/total:gc-cycles":
			checkUint64(t, name, samples[i].Value.Uint64(), uint64(mstats.NumGC))
		}
	}

	// Check tinyAllocs.
	nonTinyAllocs := uint64(0)
	for _, c := range allocsBySize.Counts {
		nonTinyAllocs += c
	}
	checkUint64(t, "/gc/heap/tiny/allocs:objects", tinyAllocs, mstats.Mallocs-nonTinyAllocs)

	// Check allocation and free counts.
	checkUint64(t, "/gc/heap/allocs:objects", mallocs, mstats.Mallocs-tinyAllocs)
	checkUint64(t, "/gc/heap/frees:objects", frees, mstats.Frees-tinyAllocs)
}

func TestReadMetricsConsistency(t *testing.T) {
	// Tests whether readMetrics produces consistent, sensible values.
	// The values are read concurrently with the runtime doing other
	// things (e.g. allocating) so what we read can't reasonably compared
	// to other runtime values (e.g. MemStats).

	// Run a few GC cycles to get some of the stats to be non-zero.
	runtime.GC()
	runtime.GC()
	runtime.GC()

	// Set GOMAXPROCS high then sleep briefly to ensure we generate
	// some idle time.
	oldmaxprocs := runtime.GOMAXPROCS(10)
	time.Sleep(time.Millisecond)
	runtime.GOMAXPROCS(oldmaxprocs)

	// Read all the supported metrics through the metrics package.
	descs, samples := prepareAllMetricsSamples()
	metrics.Read(samples)

	// Check to make sure the values we read make sense.
	var totalVirtual struct {
		got, want uint64
	}
	var objects struct {
		alloc, free             *metrics.Float64Histogram
		allocs, frees           uint64
		allocdBytes, freedBytes uint64
		total, totalBytes       uint64
	}
	var gc struct {
		numGC  uint64
		pauses uint64
	}
	var totalScan struct {
		got, want uint64
	}
	var cpu struct {
		gcAssist    float64
		gcDedicated float64
		gcIdle      float64
		gcPause     float64
		gcTotal     float64

		idle float64
		user float64

		scavengeAssist float64
		scavengeBg     float64
		scavengeTotal  float64

		total float64
	}
	for i := range samples {
		kind := samples[i].Value.Kind()
		if want := descs[samples[i].Name].Kind; kind != want {
			t.Errorf("supported metric %q has unexpected kind: got %d, want %d", samples[i].Name, kind, want)
			continue
		}
		if samples[i].Name != "/memory/classes/total:bytes" && strings.HasPrefix(samples[i].Name, "/memory/classes") {
			v := samples[i].Value.Uint64()
			totalVirtual.want += v

			// None of these stats should ever get this big.
			// If they do, there's probably overflow involved,
			// usually due to bad accounting.
			if int64(v) < 0 {
				t.Errorf("%q has high/negative value: %d", samples[i].Name, v)
			}
		}
		switch samples[i].Name {
		case "/cpu/classes/gc/mark/assist:cpu-seconds":
			cpu.gcAssist = samples[i].Value.Float64()
		case "/cpu/classes/gc/mark/dedicated:cpu-seconds":
			cpu.gcDedicated = samples[i].Value.Float64()
		case "/cpu/classes/gc/mark/idle:cpu-seconds":
			cpu.gcIdle = samples[i].Value.Float64()
		case "/cpu/classes/gc/pause:cpu-seconds":
			cpu.gcPause = samples[i].Value.Float64()
		case "/cpu/classes/gc/total:cpu-seconds":
			cpu.gcTotal = samples[i].Value.Float64()
		case "/cpu/classes/idle:cpu-seconds":
			cpu.idle = samples[i].Value.Float64()
		case "/cpu/classes/scavenge/assist:cpu-seconds":
			cpu.scavengeAssist = samples[i].Value.Float64()
		case "/cpu/classes/scavenge/background:cpu-seconds":
			cpu.scavengeBg = samples[i].Value.Float64()
		case "/cpu/classes/scavenge/total:cpu-seconds":
			cpu.scavengeTotal = samples[i].Value.Float64()
		case "/cpu/classes/total:cpu-seconds":
			cpu.total = samples[i].Value.Float64()
		case "/cpu/classes/user:cpu-seconds":
			cpu.user = samples[i].Value.Float64()
		case "/memory/classes/total:bytes":
			totalVirtual.got = samples[i].Value.Uint64()
		case "/memory/classes/heap/objects:bytes":
			objects.totalBytes = samples[i].Value.Uint64()
		case "/gc/heap/objects:objects":
			objects.total = samples[i].Value.Uint64()
		case "/gc/heap/allocs:bytes":
			objects.allocdBytes = samples[i].Value.Uint64()
		case "/gc/heap/allocs:objects":
			objects.allocs = samples[i].Value.Uint64()
		case "/gc/heap/allocs-by-size:bytes":
			objects.alloc = samples[i].Value.Float64Histogram()
		case "/gc/heap/frees:bytes":
			objects.freedBytes = samples[i].Value.Uint64()
		case "/gc/heap/frees:objects":
			objects.frees = samples[i].Value.Uint64()
		case "/gc/heap/frees-by-size:bytes":
			objects.free = samples[i].Value.Float64Histogram()
		case "/gc/cycles:gc-cycles":
			gc.numGC = samples[i].Value.Uint64()
		case "/gc/pauses:seconds":
			h := samples[i].Value.Float64Histogram()
			gc.pauses = 0
			for i := range h.Counts {
				gc.pauses += h.Counts[i]
			}
		case "/gc/scan/heap:bytes":
			totalScan.want += samples[i].Value.Uint64()
		case "/gc/scan/globals:bytes":
			totalScan.want += samples[i].Value.Uint64()
		case "/gc/scan/stack:bytes":
			totalScan.want += samples[i].Value.Uint64()
		case "/gc/scan/total:bytes":
			totalScan.got = samples[i].Value.Uint64()
		case "/sched/gomaxprocs:threads":
			if got, want := samples[i].Value.Uint64(), uint64(runtime.GOMAXPROCS(-1)); got != want {
				t.Errorf("gomaxprocs doesn't match runtime.GOMAXPROCS: got %d, want %d", got, want)
			}
		case "/sched/goroutines:goroutines":
			if samples[i].Value.Uint64() < 1 {
				t.Error("number of goroutines is less than one")
			}
		}
	}
	// Only check this on Linux where we can be reasonably sure we have a high-resolution timer.
	if runtime.GOOS == "linux" {
		if cpu.gcDedicated <= 0 && cpu.gcAssist <= 0 && cpu.gcIdle <= 0 {
			t.Errorf("found no time spent on GC work: %#v", cpu)
		}
		if cpu.gcPause <= 0 {
			t.Errorf("found no GC pauses: %f", cpu.gcPause)
		}
		if cpu.idle <= 0 {
			t.Errorf("found no idle time: %f", cpu.idle)
		}
		if total := cpu.gcDedicated + cpu.gcAssist + cpu.gcIdle + cpu.gcPause; !withinEpsilon(cpu.gcTotal, total, 0.01) {
			t.Errorf("calculated total GC CPU not within 1%% of sampled total: %f vs. %f", total, cpu.gcTotal)
		}
		if total := cpu.scavengeAssist + cpu.scavengeBg; !withinEpsilon(cpu.scavengeTotal, total, 0.01) {
			t.Errorf("calculated total scavenge CPU not within 1%% of sampled total: %f vs. %f", total, cpu.scavengeTotal)
		}
		if cpu.total <= 0 {
			t.Errorf("found no total CPU time passed")
		}
		if cpu.user <= 0 {
			t.Errorf("found no user time passed")
		}
		if total := cpu.gcTotal + cpu.scavengeTotal + cpu.user + cpu.idle; !withinEpsilon(cpu.total, total, 0.02) {
			t.Errorf("calculated total CPU not within 2%% of sampled total: %f vs. %f", total, cpu.total)
		}
	}
	if totalVirtual.got != totalVirtual.want {
		t.Errorf(`"/memory/classes/total:bytes" does not match sum of /memory/classes/**: got %d, want %d`, totalVirtual.got, totalVirtual.want)
	}
	if got, want := objects.allocs-objects.frees, objects.total; got != want {
		t.Errorf("mismatch between object alloc/free tallies and total: got %d, want %d", got, want)
	}
	if got, want := objects.allocdBytes-objects.freedBytes, objects.totalBytes; got != want {
		t.Errorf("mismatch between object alloc/free tallies and total: got %d, want %d", got, want)
	}
	if b, c := len(objects.alloc.Buckets), len(objects.alloc.Counts); b != c+1 {
		t.Errorf("allocs-by-size has wrong bucket or counts length: %d buckets, %d counts", b, c)
	}
	if b, c := len(objects.free.Buckets), len(objects.free.Counts); b != c+1 {
		t.Errorf("frees-by-size has wrong bucket or counts length: %d buckets, %d counts", b, c)
	}
	if len(objects.alloc.Buckets) != len(objects.free.Buckets) {
		t.Error("allocs-by-size and frees-by-size buckets don't match in length")
	} else if len(objects.alloc.Counts) != len(objects.free.Counts) {
		t.Error("allocs-by-size and frees-by-size counts don't match in length")
	} else {
		for i := range objects.alloc.Buckets {
			ba := objects.alloc.Buckets[i]
			bf := objects.free.Buckets[i]
			if ba != bf {
				t.Errorf("bucket %d is different for alloc and free hists: %f != %f", i, ba, bf)
			}
		}
		if !t.Failed() {
			var gotAlloc, gotFree uint64
			want := objects.total
			for i := range objects.alloc.Counts {
				if objects.alloc.Counts[i] < objects.free.Counts[i] {
					t.Errorf("found more allocs than frees in object dist bucket %d", i)
					continue
				}
				gotAlloc += objects.alloc.Counts[i]
				gotFree += objects.free.Counts[i]
			}
			if got := gotAlloc - gotFree; got != want {
				t.Errorf("object distribution counts don't match count of live objects: got %d, want %d", got, want)
			}
			if gotAlloc != objects.allocs {
				t.Errorf("object distribution counts don't match total allocs: got %d, want %d", gotAlloc, objects.allocs)
			}
			if gotFree != objects.frees {
				t.Errorf("object distribution counts don't match total allocs: got %d, want %d", gotFree, objects.frees)
			}
		}
	}
	// The current GC has at least 2 pauses per GC.
	// Check to see if that value makes sense.
	if gc.pauses < gc.numGC*2 {
		t.Errorf("fewer pauses than expected: got %d, want at least %d", gc.pauses, gc.numGC*2)
	}
	if totalScan.got <= 0 {
		t.Errorf("scannable GC space is empty: %d", totalScan.got)
	}
	if totalScan.got != totalScan.want {
		t.Errorf("/gc/scan/total:bytes doesn't line up with sum of /gc/scan*: total %d vs. sum %d", totalScan.got, totalScan.want)
	}
}

func BenchmarkReadMetricsLatency(b *testing.B) {
	stop := applyGCLoad(b)

	// Spend this much time measuring latencies.
	latencies := make([]time.Duration, 0, 1024)
	_, samples := prepareAllMetricsSamples()

	// Hit metrics.Read continuously and measure.
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		start := time.Now()
		metrics.Read(samples)
		latencies = append(latencies, time.Since(start))
	}
	// Make sure to stop the timer before we wait! The load created above
	// is very heavy-weight and not easy to stop, so we could end up
	// confusing the benchmarking framework for small b.N.
	b.StopTimer()
	stop()

	// Disable the default */op metrics.
	// ns/op doesn't mean anything because it's an average, but we
	// have a sleep in our b.N loop above which skews this significantly.
	b.ReportMetric(0, "ns/op")
	b.ReportMetric(0, "B/op")
	b.ReportMetric(0, "allocs/op")

	// Sort latencies then report percentiles.
	sort.Slice(latencies, func(i, j int) bool {
		return latencies[i] < latencies[j]
	})
	b.ReportMetric(float64(latencies[len(latencies)*50/100]), "p50-ns")
	b.ReportMetric(float64(latencies[len(latencies)*90/100]), "p90-ns")
	b.ReportMetric(float64(latencies[len(latencies)*99/100]), "p99-ns")
}

var readMetricsSink [1024]interface{}

func TestReadMetricsCumulative(t *testing.T) {
	// Set up the set of metrics marked cumulative.
	descs := metrics.All()
	var samples [2][]metrics.Sample
	samples[0] = make([]metrics.Sample, len(descs))
	samples[1] = make([]metrics.Sample, len(descs))
	total := 0
	for i := range samples[0] {
		if !descs[i].Cumulative {
			continue
		}
		samples[0][total].Name = descs[i].Name
		total++
	}
	samples[0] = samples[0][:total]
	samples[1] = samples[1][:total]
	copy(samples[1], samples[0])

	// Start some noise in the background.
	var wg sync.WaitGroup
	wg.Add(1)
	done := make(chan struct{})
	go func() {
		defer wg.Done()
		for {
			// Add more things here that could influence metrics.
			for i := 0; i < len(readMetricsSink); i++ {
				readMetricsSink[i] = make([]byte, 1024)
				select {
				case <-done:
					return
				default:
				}
			}
			runtime.GC()
		}
	}()

	sum := func(us []uint64) uint64 {
		total := uint64(0)
		for _, u := range us {
			total += u
		}
		return total
	}

	// Populate the first generation.
	metrics.Read(samples[0])

	// Check to make sure that these metrics only grow monotonically.
	for gen := 1; gen < 10; gen++ {
		metrics.Read(samples[gen%2])
		for i := range samples[gen%2] {
			name := samples[gen%2][i].Name
			vNew, vOld := samples[gen%2][i].Value, samples[1-(gen%2)][i].Value

			switch vNew.Kind() {
			case metrics.KindUint64:
				new := vNew.Uint64()
				old := vOld.Uint64()
				if new < old {
					t.Errorf("%s decreased: %d < %d", name, new, old)
				}
			case metrics.KindFloat64:
				new := vNew.Float64()
				old := vOld.Float64()
				if new < old {
					t.Errorf("%s decreased: %f < %f", name, new, old)
				}
			case metrics.KindFloat64Histogram:
				new := sum(vNew.Float64Histogram().Counts)
				old := sum(vOld.Float64Histogram().Counts)
				if new < old {
					t.Errorf("%s counts decreased: %d < %d", name, new, old)
				}
			}
		}
	}
	close(done)

	wg.Wait()
}

func withinEpsilon(v1, v2, e float64) bool {
	return v2-v2*e <= v1 && v1 <= v2+v2*e
}

func TestMutexWaitTimeMetric(t *testing.T) {
	var sample [1]metrics.Sample
	sample[0].Name = "/sync/mutex/wait/total:seconds"

	locks := []locker2{
		new(mutex),
		new(rwmutexWrite),
		new(rwmutexReadWrite),
		new(rwmutexWriteRead),
	}
	for _, lock := range locks {
		t.Run(reflect.TypeOf(lock).Elem().Name(), func(t *testing.T) {
			metrics.Read(sample[:])
			before := time.Duration(sample[0].Value.Float64() * 1e9)

			minMutexWaitTime := generateMutexWaitTime(lock)

			metrics.Read(sample[:])
			after := time.Duration(sample[0].Value.Float64() * 1e9)

			if wt := after - before; wt < minMutexWaitTime {
				t.Errorf("too little mutex wait time: got %s, want %s", wt, minMutexWaitTime)
			}
		})
	}
}

// locker2 represents an API surface of two concurrent goroutines
// locking the same resource, but through different APIs. It's intended
// to abstract over the relationship of two Lock calls or an RLock
// and a Lock call.
type locker2 interface {
	Lock1()
	Unlock1()
	Lock2()
	Unlock2()
}

type mutex struct {
	mu sync.Mutex
}

func (m *mutex) Lock1()   { m.mu.Lock() }
func (m *mutex) Unlock1() { m.mu.Unlock() }
func (m *mutex) Lock2()   { m.mu.Lock() }
func (m *mutex) Unlock2() { m.mu.Unlock() }

type rwmutexWrite struct {
	mu sync.RWMutex
}

func (m *rwmutexWrite) Lock1()   { m.mu.Lock() }
func (m *rwmutexWrite) Unlock1() { m.mu.Unlock() }
func (m *rwmutexWrite) Lock2()   { m.mu.Lock() }
func (m *rwmutexWrite) Unlock2() { m.mu.Unlock() }

type rwmutexReadWrite struct {
	mu sync.RWMutex
}

func (m *rwmutexReadWrite) Lock1()   { m.mu.RLock() }
func (m *rwmutexReadWrite) Unlock1() { m.mu.RUnlock() }
func (m *rwmutexReadWrite) Lock2()   { m.mu.Lock() }
func (m *rwmutexReadWrite) Unlock2() { m.mu.Unlock() }

type rwmutexWriteRead struct {
	mu sync.RWMutex
}

func (m *rwmutexWriteRead) Lock1()   { m.mu.Lock() }
func (m *rwmutexWriteRead) Unlock1() { m.mu.Unlock() }
func (m *rwmutexWriteRead) Lock2()   { m.mu.RLock() }
func (m *rwmutexWriteRead) Unlock2() { m.mu.RUnlock() }

// generateMutexWaitTime causes a couple of goroutines
// to block a whole bunch of times on a sync.Mutex, returning
// the minimum amount of time that should be visible in the
// /sync/mutex-wait:seconds metric.
func generateMutexWaitTime(mu locker2) time.Duration {
	// Set up the runtime to always track casgstatus transitions for metrics.
	*runtime.CasGStatusAlwaysTrack = true

	mu.Lock1()

	// Start up a goroutine to wait on the lock.
	gc := make(chan *runtime.G)
	done := make(chan bool)
	go func() {
		gc <- runtime.Getg()

		for {
			mu.Lock2()
			mu.Unlock2()
			if <-done {
				return
			}
		}
	}()
	gp := <-gc

	// Set the block time high enough so that it will always show up, even
	// on systems with coarse timer granularity.
	const blockTime = 100 * time.Millisecond

	// Make sure the goroutine spawned above actually blocks on the lock.
	for {
		if runtime.GIsWaitingOnMutex(gp) {
			break
		}
		runtime.Gosched()
	}

	// Let some amount of time pass.
	time.Sleep(blockTime)

	// Let the other goroutine acquire the lock.
	mu.Unlock1()
	done <- true

	// Reset flag.
	*runtime.CasGStatusAlwaysTrack = false
	return blockTime
}

// See issue #60276.
func TestCPUMetricsSleep(t *testing.T) {
	if runtime.GOOS == "wasip1" {
		// Since wasip1 busy-waits in the scheduler, there's no meaningful idle
		// time. This is accurately reflected in the metrics, but it means this
		// test is basically meaningless on this platform.
		t.Skip("wasip1 currently busy-waits in idle time; test not applicable")
	}

	names := []string{
		"/cpu/classes/idle:cpu-seconds",

		"/cpu/classes/gc/mark/assist:cpu-seconds",
		"/cpu/classes/gc/mark/dedicated:cpu-seconds",
		"/cpu/classes/gc/mark/idle:cpu-seconds",
		"/cpu/classes/gc/pause:cpu-seconds",
		"/cpu/classes/gc/total:cpu-seconds",
		"/cpu/classes/scavenge/assist:cpu-seconds",
		"/cpu/classes/scavenge/background:cpu-seconds",
		"/cpu/classes/scavenge/total:cpu-seconds",
		"/cpu/classes/total:cpu-seconds",
		"/cpu/classes/user:cpu-seconds",
	}
	prep := func() []metrics.Sample {
		mm := make([]metrics.Sample, len(names))
		for i := range names {
			mm[i].Name = names[i]
		}
		return mm
	}
	m1, m2 := prep(), prep()

	const (
		// Expected time spent idle.
		dur = 100 * time.Millisecond

		// maxFailures is the number of consecutive failures requires to cause the test to fail.
		maxFailures = 10
	)

	failureIdleTimes := make([]float64, 0, maxFailures)

	// If the bug we expect is happening, then the Sleep CPU time will be accounted for
	// as user time rather than idle time. In an ideal world we'd expect the whole application
	// to go instantly idle the moment this goroutine goes to sleep, and stay asleep for that
	// duration. However, the Go runtime can easily eat into idle time while this goroutine is
	// blocked in a sleep. For example, slow platforms might spend more time expected in the
	// scheduler. Another example is that a Go runtime background goroutine could run while
	// everything else is idle. Lastly, if a running goroutine is descheduled by the OS, enough
	// time may pass such that the goroutine is ready to wake, even though the runtime couldn't
	// observe itself as idle with nanotime.
	//
	// To deal with all this, we give a half-proc's worth of leniency.
	//
	// We also retry multiple times to deal with the fact that the OS might deschedule us before
	// we yield and go idle. That has a rare enough chance that retries should resolve it.
	// If the issue we expect is happening, it should be persistent.
	minIdleCPUSeconds := dur.Seconds() * (float64(runtime.GOMAXPROCS(-1)) - 0.5)

	// Let's make sure there's no background scavenge work to do.
	//
	// The runtime.GC calls below ensure the background sweeper
	// will not run during the idle period.
	debug.FreeOSMemory()

	for retries := 0; retries < maxFailures; retries++ {
		// Read 1.
		runtime.GC() // Update /cpu/classes metrics.
		metrics.Read(m1)

		// Sleep.
		time.Sleep(dur)

		// Read 2.
		runtime.GC() // Update /cpu/classes metrics.
		metrics.Read(m2)

		dt := m2[0].Value.Float64() - m1[0].Value.Float64()
		if dt >= minIdleCPUSeconds {
			// All is well. Test passed.
			return
		}
		failureIdleTimes = append(failureIdleTimes, dt)
		// Try again.
	}

	// We couldn't observe the expected idle time even once.
	for i, dt := range failureIdleTimes {
		t.Logf("try %2d: idle time = %.5fs\n", i+1, dt)
	}
	t.Logf("try %d breakdown:\n", len(failureIdleTimes))
	for i := range names {
		if m1[i].Value.Kind() == metrics.KindBad {
			continue
		}
		t.Logf("\t%s %0.3f\n", names[i], m2[i].Value.Float64()-m1[i].Value.Float64())
	}
	t.Errorf(`time.Sleep did not contribute enough to "idle" class: minimum idle time = %.5fs`, minIdleCPUSeconds)
}

func TestMetricHeapUnusedLargeObjectOverflow(t *testing.T) {
	// This test makes sure /memory/classes/heap/unused:bytes
	// doesn't overflow when allocating and deallocating large
	// objects. It is a regression test for #67019.
	done := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			for i := 0; i < 10; i++ {
				runtime.Escape(make([]byte, 1<<20))
			}
			runtime.GC()
			select {
			case <-done:
				return
			default:
			}
		}
	}()
	s := []metrics.Sample{
		{Name: "/memory/classes/heap/unused:bytes"},
	}
	for i := 0; i < 1000; i++ {
		metrics.Read(s)
		if s[0].Value.Uint64() > 1<<40 {
			t.Errorf("overflow")
			break
		}
	}
	done <- struct{}{}
	wg.Wait()
}

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"runtime"
	"runtime/metrics"
	"sort"
	"strings"
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
		case "/gc/heap/objects:objects":
			checkUint64(t, name, samples[i].Value.Uint64(), mstats.HeapObjects)
		case "/gc/heap/goal:bytes":
			checkUint64(t, name, samples[i].Value.Uint64(), mstats.NextGC)
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
	// to runtime values.

	// Run a few GC cycles to get some of the stats to be non-zero.
	runtime.GC()
	runtime.GC()
	runtime.GC()

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
		case "/sched/goroutines:goroutines":
			if samples[i].Value.Uint64() < 1 {
				t.Error("number of goroutines is less than one")
			}
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
		latencies = append(latencies, time.Now().Sub(start))
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

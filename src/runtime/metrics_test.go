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
		}
	}
}

func TestReadMetricsConsistency(t *testing.T) {
	// Tests whether readMetrics produces consistent, sensible values.
	// The values are read concurrently with the runtime doing other
	// things (e.g. allocating) so what we read can't reasonably compared
	// to runtime values.

	// Read all the supported metrics through the metrics package.
	descs, samples := prepareAllMetricsSamples()
	metrics.Read(samples)

	// Check to make sure the values we read make sense.
	var totalVirtual struct {
		got, want uint64
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
		}
	}
	if totalVirtual.got != totalVirtual.want {
		t.Errorf(`"/memory/classes/total:bytes" does not match sum of /memory/classes/**: got %d, want %d`, totalVirtual.got, totalVirtual.want)
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

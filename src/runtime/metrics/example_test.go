// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package metrics_test

import (
	"fmt"
	"runtime/metrics"
)

func ExampleRead_readingOneMetric() {
	// Name of the metric we want to read.
	const myMetric = "/memory/classes/heap/free:bytes"

	// Create a sample for the metric.
	sample := make([]metrics.Sample, 1)
	sample[0].Name = myMetric

	// Sample the metric.
	metrics.Read(sample)

	// Check if the metric is actually supported.
	// If it's not, the resulting value will always have
	// kind KindBad.
	if sample[0].Value.Kind() == metrics.KindBad {
		panic(fmt.Sprintf("metric %q no longer supported", myMetric))
	}

	// Handle the result.
	//
	// It's OK to assume a particular Kind for a metric;
	// they're guaranteed not to change.
	freeBytes := sample[0].Value.Uint64()

	fmt.Printf("free but not released memory: %d\n", freeBytes)
}

func ExampleRead_readingAllMetrics() {
	// Get descriptions for all supported metrics.
	descs := metrics.All()

	// Create a sample for each metric.
	samples := make([]metrics.Sample, len(descs))
	for i := range samples {
		samples[i].Name = descs[i].Name
	}

	// Sample the metrics. Re-use the samples slice if you can!
	metrics.Read(samples)

	// Iterate over all results.
	for _, sample := range samples {
		// Pull out the name and value.
		name, value := sample.Name, sample.Value

		// Handle each sample.
		switch value.Kind() {
		case metrics.KindUint64:
			fmt.Printf("%s: %d\n", name, value.Uint64())
		case metrics.KindFloat64:
			fmt.Printf("%s: %f\n", name, value.Float64())
		case metrics.KindFloat64Histogram:
			// The histogram may be quite large, so let's just pull out
			// a crude estimate for the median for the sake of this example.
			fmt.Printf("%s: %f\n", name, medianBucket(value.Float64Histogram()))
		case metrics.KindBad:
			// This should never happen because all metrics are supported
			// by construction.
			panic("bug in runtime/metrics package!")
		default:
			// This may happen as new metrics get added.
			//
			// The safest thing to do here is to simply log it somewhere
			// as something to look into, but ignore it for now.
			// In the worst case, you might temporarily miss out on a new metric.
			fmt.Printf("%s: unexpected metric Kind: %v\n", name, value.Kind())
		}
	}
}

func medianBucket(h *metrics.Float64Histogram) float64 {
	total := uint64(0)
	for _, count := range h.Counts {
		total += count
	}
	thresh := total / 2
	total = 0
	for i, count := range h.Counts {
		total += count
		if total >= thresh {
			return h.Buckets[i]
		}
	}
	panic("should not happen")
}

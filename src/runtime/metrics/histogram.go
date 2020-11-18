// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package metrics

// Float64Histogram represents a distribution of float64 values.
type Float64Histogram struct {
	// Counts contains the weights for each histogram bucket. The length of
	// Counts is equal to the length of Buckets (in the metric description)
	// plus one to account for the implicit minimum bucket.
	//
	// Given N buckets, the following is the mathematical relationship between
	// Counts and Buckets.
	// count[0] is the weight of the range (-inf, bucket[0])
	// count[n] is the weight of the range [bucket[n], bucket[n+1]), for 0 < n < N-1
	// count[N-1] is the weight of the range [bucket[N-1], inf)
	Counts []uint64

	// Buckets contains the boundaries between histogram buckets, in increasing order.
	//
	// Because this slice contains boundaries, there are len(Buckets)+1 counts:
	// a count for all values less than the first boundary, a count covering each
	// [slice[i], slice[i+1]) interval, and a count for all values greater than or
	// equal to the last boundary.
	//
	// For a given metric name, the value of Buckets is guaranteed not to change
	// between calls until program exit.
	Buckets []float64
}

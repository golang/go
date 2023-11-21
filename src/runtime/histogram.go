// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"runtime/internal/atomic"
	"runtime/internal/sys"
	"unsafe"
)

const (
	// For the time histogram type, we use an HDR histogram.
	// Values are placed in buckets based solely on the most
	// significant set bit. Thus, buckets are power-of-2 sized.
	// Values are then placed into sub-buckets based on the value of
	// the next timeHistSubBucketBits most significant bits. Thus,
	// sub-buckets are linear within a bucket.
	//
	// Therefore, the number of sub-buckets (timeHistNumSubBuckets)
	// defines the error. This error may be computed as
	// 1/timeHistNumSubBuckets*100%. For example, for 16 sub-buckets
	// per bucket the error is approximately 6%.
	//
	// The number of buckets (timeHistNumBuckets), on the
	// other hand, defines the range. To avoid producing a large number
	// of buckets that are close together, especially for small numbers
	// (e.g. 1, 2, 3, 4, 5 ns) that aren't very useful, timeHistNumBuckets
	// is defined in terms of the least significant bit (timeHistMinBucketBits)
	// that needs to be set before we start bucketing and the most
	// significant bit (timeHistMaxBucketBits) that we bucket before we just
	// dump it into a catch-all bucket.
	//
	// As an example, consider the configuration:
	//
	//    timeHistMinBucketBits = 9
	//    timeHistMaxBucketBits = 48
	//    timeHistSubBucketBits = 2
	//
	// Then:
	//
	//    011000001
	//    ^--
	//    │ ^
	//    │ └---- Next 2 bits -> sub-bucket 3
	//    └------- Bit 9 unset -> bucket 0
	//
	//    110000001
	//    ^--
	//    │ ^
	//    │ └---- Next 2 bits -> sub-bucket 2
	//    └------- Bit 9 set -> bucket 1
	//
	//    1000000010
	//    ^-- ^
	//    │ ^ └-- Lower bits ignored
	//    │ └---- Next 2 bits -> sub-bucket 0
	//    └------- Bit 10 set -> bucket 2
	//
	// Following this pattern, bucket 38 will have the bit 46 set. We don't
	// have any buckets for higher values, so we spill the rest into an overflow
	// bucket containing values of 2^47-1 nanoseconds or approx. 1 day or more.
	// This range is more than enough to handle durations produced by the runtime.
	timeHistMinBucketBits = 9
	timeHistMaxBucketBits = 48 // Note that this is exclusive; 1 higher than the actual range.
	timeHistSubBucketBits = 2
	timeHistNumSubBuckets = 1 << timeHistSubBucketBits
	timeHistNumBuckets    = timeHistMaxBucketBits - timeHistMinBucketBits + 1
	// Two extra buckets, one for underflow, one for overflow.
	timeHistTotalBuckets = timeHistNumBuckets*timeHistNumSubBuckets + 2
)

// timeHistogram represents a distribution of durations in
// nanoseconds.
//
// The accuracy and range of the histogram is defined by the
// timeHistSubBucketBits and timeHistNumBuckets constants.
//
// It is an HDR histogram with exponentially-distributed
// buckets and linearly distributed sub-buckets.
//
// The histogram is safe for concurrent reads and writes.
type timeHistogram struct {
	counts [timeHistNumBuckets * timeHistNumSubBuckets]atomic.Uint64

	// underflow counts all the times we got a negative duration
	// sample. Because of how time works on some platforms, it's
	// possible to measure negative durations. We could ignore them,
	// but we record them anyway because it's better to have some
	// signal that it's happening than just missing samples.
	underflow atomic.Uint64

	// overflow counts all the times we got a duration that exceeded
	// the range counts represents.
	overflow atomic.Uint64
}

// record adds the given duration to the distribution.
//
// Disallow preemptions and stack growths because this function
// may run in sensitive locations.
//
//go:nosplit
func (h *timeHistogram) record(duration int64) {
	// If the duration is negative, capture that in underflow.
	if duration < 0 {
		h.underflow.Add(1)
		return
	}
	// bucketBit is the target bit for the bucket which is usually the
	// highest 1 bit, but if we're less than the minimum, is the highest
	// 1 bit of the minimum (which will be zero in the duration).
	//
	// bucket is the bucket index, which is the bucketBit minus the
	// highest bit of the minimum, plus one to leave room for the catch-all
	// bucket for samples lower than the minimum.
	var bucketBit, bucket uint
	if l := sys.Len64(uint64(duration)); l < timeHistMinBucketBits {
		bucketBit = timeHistMinBucketBits
		bucket = 0 // bucketBit - timeHistMinBucketBits
	} else {
		bucketBit = uint(l)
		bucket = bucketBit - timeHistMinBucketBits + 1
	}
	// If the bucket we computed is greater than the number of buckets,
	// count that in overflow.
	if bucket >= timeHistNumBuckets {
		h.overflow.Add(1)
		return
	}
	// The sub-bucket index is just next timeHistSubBucketBits after the bucketBit.
	subBucket := uint(duration>>(bucketBit-1-timeHistSubBucketBits)) % timeHistNumSubBuckets
	h.counts[bucket*timeHistNumSubBuckets+subBucket].Add(1)
}

// write dumps the histogram to the passed metricValue as a float64 histogram.
func (h *timeHistogram) write(out *metricValue) {
	hist := out.float64HistOrInit(timeHistBuckets)
	// The bottom-most bucket, containing negative values, is tracked
	// separately as underflow, so fill that in manually and then iterate
	// over the rest.
	hist.counts[0] = h.underflow.Load()
	for i := range h.counts {
		hist.counts[i+1] = h.counts[i].Load()
	}
	hist.counts[len(hist.counts)-1] = h.overflow.Load()
}

const (
	fInf    = 0x7FF0000000000000
	fNegInf = 0xFFF0000000000000
)

func float64Inf() float64 {
	inf := uint64(fInf)
	return *(*float64)(unsafe.Pointer(&inf))
}

func float64NegInf() float64 {
	inf := uint64(fNegInf)
	return *(*float64)(unsafe.Pointer(&inf))
}

// timeHistogramMetricsBuckets generates a slice of boundaries for
// the timeHistogram. These boundaries are represented in seconds,
// not nanoseconds like the timeHistogram represents durations.
func timeHistogramMetricsBuckets() []float64 {
	b := make([]float64, timeHistTotalBuckets+1)
	// Underflow bucket.
	b[0] = float64NegInf()

	for j := 0; j < timeHistNumSubBuckets; j++ {
		// No bucket bit for the first few buckets. Just sub-bucket bits after the
		// min bucket bit.
		bucketNanos := uint64(j) << (timeHistMinBucketBits - 1 - timeHistSubBucketBits)
		// Convert nanoseconds to seconds via a division.
		// These values will all be exactly representable by a float64.
		b[j+1] = float64(bucketNanos) / 1e9
	}
	// Generate the rest of the buckets. It's easier to reason
	// about if we cut out the 0'th bucket.
	for i := timeHistMinBucketBits; i < timeHistMaxBucketBits; i++ {
		for j := 0; j < timeHistNumSubBuckets; j++ {
			// Set the bucket bit.
			bucketNanos := uint64(1) << (i - 1)
			// Set the sub-bucket bits.
			bucketNanos |= uint64(j) << (i - 1 - timeHistSubBucketBits)
			// The index for this bucket is going to be the (i+1)'th bucket
			// (note that we're starting from zero, but handled the first bucket
			// earlier, so we need to compensate), and the j'th sub bucket.
			// Add 1 because we left space for -Inf.
			bucketIndex := (i-timeHistMinBucketBits+1)*timeHistNumSubBuckets + j + 1
			// Convert nanoseconds to seconds via a division.
			// These values will all be exactly representable by a float64.
			b[bucketIndex] = float64(bucketNanos) / 1e9
		}
	}
	// Overflow bucket.
	b[len(b)-2] = float64(uint64(1)<<(timeHistMaxBucketBits-1)) / 1e9
	b[len(b)-1] = float64Inf()
	return b
}

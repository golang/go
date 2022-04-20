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
	// Values are placed in super-buckets based solely on the most
	// significant set bit. Thus, super-buckets are power-of-2 sized.
	// Values are then placed into sub-buckets based on the value of
	// the next timeHistSubBucketBits most significant bits. Thus,
	// sub-buckets are linear within a super-bucket.
	//
	// Therefore, the number of sub-buckets (timeHistNumSubBuckets)
	// defines the error. This error may be computed as
	// 1/timeHistNumSubBuckets*100%. For example, for 16 sub-buckets
	// per super-bucket the error is approximately 6%.
	//
	// The number of super-buckets (timeHistNumSuperBuckets), on the
	// other hand, defines the range. To reserve room for sub-buckets,
	// bit timeHistSubBucketBits is the first bit considered for
	// super-buckets, so super-bucket indices are adjusted accordingly.
	//
	// As an example, consider 45 super-buckets with 16 sub-buckets.
	//
	//    00110
	//    ^----
	//    │  ^
	//    │  └---- Lowest 4 bits -> sub-bucket 6
	//    └------- Bit 4 unset -> super-bucket 0
	//
	//    10110
	//    ^----
	//    │  ^
	//    │  └---- Next 4 bits -> sub-bucket 6
	//    └------- Bit 4 set -> super-bucket 1
	//    100010
	//    ^----^
	//    │  ^ └-- Lower bits ignored
	//    │  └---- Next 4 bits -> sub-bucket 1
	//    └------- Bit 5 set -> super-bucket 2
	//
	// Following this pattern, super-bucket 44 will have the bit 47 set. We don't
	// have any buckets for higher values, so the highest sub-bucket will
	// contain values of 2^48-1 nanoseconds or approx. 3 days. This range is
	// more than enough to handle durations produced by the runtime.
	timeHistSubBucketBits   = 4
	timeHistNumSubBuckets   = 1 << timeHistSubBucketBits
	timeHistNumSuperBuckets = 45
	timeHistTotalBuckets    = timeHistNumSuperBuckets*timeHistNumSubBuckets + 1
)

// timeHistogram represents a distribution of durations in
// nanoseconds.
//
// The accuracy and range of the histogram is defined by the
// timeHistSubBucketBits and timeHistNumSuperBuckets constants.
//
// It is an HDR histogram with exponentially-distributed
// buckets and linearly distributed sub-buckets.
//
// Counts in the histogram are updated atomically, so it is safe
// for concurrent use. It is also safe to read all the values
// atomically.
type timeHistogram struct {
	counts [timeHistNumSuperBuckets * timeHistNumSubBuckets]uint64

	// underflow counts all the times we got a negative duration
	// sample. Because of how time works on some platforms, it's
	// possible to measure negative durations. We could ignore them,
	// but we record them anyway because it's better to have some
	// signal that it's happening than just missing samples.
	underflow uint64
}

// record adds the given duration to the distribution.
//
// Disallow preemptions and stack growths because this function
// may run in sensitive locations.
//
//go:nosplit
func (h *timeHistogram) record(duration int64) {
	if duration < 0 {
		atomic.Xadd64(&h.underflow, 1)
		return
	}
	// The index of the exponential bucket is just the index
	// of the highest set bit adjusted for how many bits we
	// use for the subbucket. Note that it's timeHistSubBucketsBits-1
	// because we use the 0th bucket to hold values < timeHistNumSubBuckets.
	var superBucket, subBucket uint
	if duration >= timeHistNumSubBuckets {
		// At this point, we know the duration value will always be
		// at least timeHistSubBucketsBits long.
		superBucket = uint(sys.Len64(uint64(duration))) - timeHistSubBucketBits
		if superBucket*timeHistNumSubBuckets >= uint(len(h.counts)) {
			// The bucket index we got is larger than what we support, so
			// include this count in the highest bucket, which extends to
			// infinity.
			superBucket = timeHistNumSuperBuckets - 1
			subBucket = timeHistNumSubBuckets - 1
		} else {
			// The linear subbucket index is just the timeHistSubBucketsBits
			// bits after the top bit. To extract that value, shift down
			// the duration such that we leave the top bit and the next bits
			// intact, then extract the index.
			subBucket = uint((duration >> (superBucket - 1)) % timeHistNumSubBuckets)
		}
	} else {
		subBucket = uint(duration)
	}
	atomic.Xadd64(&h.counts[superBucket*timeHistNumSubBuckets+subBucket], 1)
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
	b[0] = float64NegInf()
	// Super-bucket 0 has no bits above timeHistSubBucketBits
	// set, so just iterate over each bucket and assign the
	// incrementing bucket.
	for i := 0; i < timeHistNumSubBuckets; i++ {
		bucketNanos := uint64(i)
		b[i+1] = float64(bucketNanos) / 1e9
	}
	// Generate the rest of the super-buckets. It's easier to reason
	// about if we cut out the 0'th bucket, so subtract one since
	// we just handled that bucket.
	for i := 0; i < timeHistNumSuperBuckets-1; i++ {
		for j := 0; j < timeHistNumSubBuckets; j++ {
			// Set the super-bucket bit.
			bucketNanos := uint64(1) << (i + timeHistSubBucketBits)
			// Set the sub-bucket bits.
			bucketNanos |= uint64(j) << i
			// The index for this bucket is going to be the (i+1)'th super bucket
			// (note that we're starting from zero, but handled the first super-bucket
			// earlier, so we need to compensate), and the j'th sub bucket.
			// Add 1 because we left space for -Inf.
			bucketIndex := (i+1)*timeHistNumSubBuckets + j + 1
			// Convert nanoseconds to seconds via a division.
			// These values will all be exactly representable by a float64.
			b[bucketIndex] = float64(bucketNanos) / 1e9
		}
	}
	b[len(b)-1] = float64Inf()
	return b
}

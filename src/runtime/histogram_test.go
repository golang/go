// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"math"
	. "runtime"
	"testing"
)

var dummyTimeHistogram TimeHistogram

func TestTimeHistogram(t *testing.T) {
	// We need to use a global dummy because this
	// could get stack-allocated with a non-8-byte alignment.
	// The result of this bad alignment is a segfault on
	// 32-bit platforms when calling Record.
	h := &dummyTimeHistogram

	// Record exactly one sample in each bucket.
	for j := 0; j < TimeHistNumSubBuckets; j++ {
		v := int64(j) << (TimeHistMinBucketBits - 1 - TimeHistSubBucketBits)
		for k := 0; k < j; k++ {
			// Record a number of times equal to the bucket index.
			h.Record(v)
		}
	}
	for i := TimeHistMinBucketBits; i < TimeHistMaxBucketBits; i++ {
		base := int64(1) << (i - 1)
		for j := 0; j < TimeHistNumSubBuckets; j++ {
			v := int64(j) << (i - 1 - TimeHistSubBucketBits)
			for k := 0; k < (i+1-TimeHistMinBucketBits)*TimeHistNumSubBuckets+j; k++ {
				// Record a number of times equal to the bucket index.
				h.Record(base + v)
			}
		}
	}
	// Hit the underflow and overflow buckets.
	h.Record(int64(-1))
	h.Record(math.MaxInt64)
	h.Record(math.MaxInt64)

	// Check to make sure there's exactly one count in each
	// bucket.
	for i := 0; i < TimeHistNumBuckets; i++ {
		for j := 0; j < TimeHistNumSubBuckets; j++ {
			c, ok := h.Count(i, j)
			if !ok {
				t.Errorf("unexpected invalid bucket: (%d, %d)", i, j)
			} else if idx := uint64(i*TimeHistNumSubBuckets + j); c != idx {
				t.Errorf("bucket (%d, %d) has count that is not %d: %d", i, j, idx, c)
			}
		}
	}
	c, ok := h.Count(-1, 0)
	if ok {
		t.Errorf("expected to hit underflow bucket: (%d, %d)", -1, 0)
	}
	if c != 1 {
		t.Errorf("overflow bucket has count that is not 1: %d", c)
	}

	c, ok = h.Count(TimeHistNumBuckets+1, 0)
	if ok {
		t.Errorf("expected to hit overflow bucket: (%d, %d)", TimeHistNumBuckets+1, 0)
	}
	if c != 2 {
		t.Errorf("overflow bucket has count that is not 2: %d", c)
	}

	dummyTimeHistogram = TimeHistogram{}
}

func TestTimeHistogramMetricsBuckets(t *testing.T) {
	buckets := TimeHistogramMetricsBuckets()

	nonInfBucketsLen := TimeHistNumSubBuckets * TimeHistNumBuckets
	expBucketsLen := nonInfBucketsLen + 3 // Count -Inf, the edge for the overflow bucket, and +Inf.
	if len(buckets) != expBucketsLen {
		t.Fatalf("unexpected length of buckets: got %d, want %d", len(buckets), expBucketsLen)
	}
	// Check some values.
	idxToBucket := map[int]float64{
		0:                 math.Inf(-1),
		1:                 0.0,
		2:                 float64(0x040) / 1e9,
		3:                 float64(0x080) / 1e9,
		4:                 float64(0x0c0) / 1e9,
		5:                 float64(0x100) / 1e9,
		6:                 float64(0x140) / 1e9,
		7:                 float64(0x180) / 1e9,
		8:                 float64(0x1c0) / 1e9,
		9:                 float64(0x200) / 1e9,
		10:                float64(0x280) / 1e9,
		11:                float64(0x300) / 1e9,
		12:                float64(0x380) / 1e9,
		13:                float64(0x400) / 1e9,
		15:                float64(0x600) / 1e9,
		81:                float64(0x8000000) / 1e9,
		82:                float64(0xa000000) / 1e9,
		108:               float64(0x380000000) / 1e9,
		expBucketsLen - 2: float64(0x1<<47) / 1e9,
		expBucketsLen - 1: math.Inf(1),
	}
	for idx, bucket := range idxToBucket {
		if got, want := buckets[idx], bucket; got != want {
			t.Errorf("expected bucket %d to have value %e, got %e", idx, want, got)
		}
	}
}

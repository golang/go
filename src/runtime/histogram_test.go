// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
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
	for i := 0; i < TimeHistNumSuperBuckets; i++ {
		var base int64
		if i > 0 {
			base = int64(1) << (i + TimeHistSubBucketBits - 1)
		}
		for j := 0; j < TimeHistNumSubBuckets; j++ {
			v := int64(j)
			if i > 0 {
				v <<= i - 1
			}
			h.Record(base + v)
		}
	}
	// Hit the overflow bucket.
	h.Record(int64(^uint64(0) >> 1))

	// Check to make sure there's exactly one count in each
	// bucket.
	for i := uint(0); i < TimeHistNumSuperBuckets; i++ {
		for j := uint(0); j < TimeHistNumSubBuckets; j++ {
			c, ok := h.Count(i, j)
			if !ok {
				t.Errorf("hit overflow bucket unexpectedly: (%d, %d)", i, j)
			} else if c != 1 {
				t.Errorf("bucket (%d, %d) has count that is not 1: %d", i, j, c)
			}
		}
	}
	c, ok := h.Count(TimeHistNumSuperBuckets, 0)
	if ok {
		t.Errorf("expected to hit overflow bucket: (%d, %d)", TimeHistNumSuperBuckets, 0)
	}
	if c != 1 {
		t.Errorf("overflow bucket has count that is not 1: %d", c)
	}
	dummyTimeHistogram = TimeHistogram{}
}

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pprof

import (
	"internal/profilerecord"
	"io"
	"math"
	"runtime"
	"strings"
)

// writeHeapProto writes the current heap profile in protobuf format to w.
func writeHeapProto(w io.Writer, p []profilerecord.MemProfileRecord, rate int64, defaultSampleType string) error {
	b := newProfileBuilder(w)
	b.pbValueType(tagProfile_PeriodType, "space", "bytes")
	b.pb.int64Opt(tagProfile_Period, rate)
	b.pbValueType(tagProfile_SampleType, "alloc_objects", "count")
	b.pbValueType(tagProfile_SampleType, "alloc_space", "bytes")
	b.pbValueType(tagProfile_SampleType, "inuse_objects", "count")
	b.pbValueType(tagProfile_SampleType, "inuse_space", "bytes")
	if defaultSampleType != "" {
		b.pb.int64Opt(tagProfile_DefaultSampleType, b.stringIndex(defaultSampleType))
	}

	values := []int64{0, 0, 0, 0}
	var locs []uint64
	for _, r := range p {
		locs = b.appendLocsForStackWithoutRuntime(locs[:0], r.Stack)

		values[0], values[1] = scaleHeapSample(r.AllocObjects, r.ObjectSize, rate)
		values[2], values[3] = scaleHeapSample(r.InUseObjects(), r.ObjectSize, rate)
		b.pbSample(values, locs, func() {
			b.pbLabel(tagSample_Label, "bytes", "", r.ObjectSize)
		})
	}
	return b.build()
}

// appendLocsForStackWithoutRuntime appends locations for a stack, hiding
// runtime leaf frames unless the stack contains only runtime frames.
func (b *profileBuilder) appendLocsForStackWithoutRuntime(locs []uint64, stk []uintptr) []uint64 {
	origStk := stk
	for i, addr := range stk {
		if f := runtime.FuncForPC(addr); f != nil && (strings.HasPrefix(f.Name(), "runtime.") || strings.HasPrefix(f.Name(), "internal/runtime/")) {
			continue
		}
		stk = stk[i:]
		break
	}
	n := len(locs)
	locs = b.appendLocsForStack(locs, stk)
	if len(locs) == n {
		// The stack was empty after filtering out runtime leaf frames,
		// so just add the original stack
		locs = b.appendLocsForStack(locs, origStk)
	}
	return locs
}

// scaleHeapSample adjusts the data from a heap Sample to
// account for its probability of appearing in the collected
// data. heap profiles are a sampling of the memory allocations
// requests in a program. We estimate the unsampled value by dividing
// each collected sample by its probability of appearing in the
// profile. heap profiles rely on a poisson process to determine
// which samples to collect, based on the desired average collection
// rate R. The probability of a sample of size S to appear in that
// profile is 1-exp(-S/R).
func scaleHeapSample(count, avgSize, rate int64) (int64, int64) {
	if count == 0 || avgSize == 0 {
		return 0, 0
	}

	if rate <= 1 {
		// if rate==1 all samples were collected so no adjustment is needed.
		// if rate<1 treat as unknown and skip scaling.
		return count, count * avgSize
	}

	scale := 1 / (1 - math.Exp(-float64(avgSize)/float64(rate)))

	return int64(float64(count) * scale), int64(float64(count*avgSize) * scale)
}

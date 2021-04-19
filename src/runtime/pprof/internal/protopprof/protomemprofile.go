// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package protopprof

import (
	"internal/pprof/profile"
	"math"
	"runtime"
	"time"
)

// EncodeMemProfile converts MemProfileRecords to a Profile.
func EncodeMemProfile(mr []runtime.MemProfileRecord, rate int64, t time.Time) *profile.Profile {
	p := &profile.Profile{
		Period:     rate,
		PeriodType: &profile.ValueType{Type: "space", Unit: "bytes"},
		SampleType: []*profile.ValueType{
			{Type: "alloc_objects", Unit: "count"},
			{Type: "alloc_space", Unit: "bytes"},
			{Type: "inuse_objects", Unit: "count"},
			{Type: "inuse_space", Unit: "bytes"},
		},
		TimeNanos: int64(t.UnixNano()),
	}

	locs := make(map[uintptr]*profile.Location)
	for _, r := range mr {
		stack := r.Stack()
		sloc := make([]*profile.Location, len(stack))
		for i, addr := range stack {
			loc := locs[addr]
			if loc == nil {
				loc = &profile.Location{
					ID:      uint64(len(p.Location) + 1),
					Address: uint64(addr),
				}
				locs[addr] = loc
				p.Location = append(p.Location, loc)
			}
			sloc[i] = loc
		}

		ao, ab := scaleHeapSample(r.AllocObjects, r.AllocBytes, rate)
		uo, ub := scaleHeapSample(r.InUseObjects(), r.InUseBytes(), rate)

		p.Sample = append(p.Sample, &profile.Sample{
			Value:    []int64{ao, ab, uo, ub},
			Location: sloc,
		})
	}
	if runtime.GOOS == "linux" {
		addMappings(p)
	}
	return p
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
func scaleHeapSample(count, size, rate int64) (int64, int64) {
	if count == 0 || size == 0 {
		return 0, 0
	}

	if rate <= 1 {
		// if rate==1 all samples were collected so no adjustment is needed.
		// if rate<1 treat as unknown and skip scaling.
		return count, size
	}

	avgSize := float64(size) / float64(count)
	scale := 1 / (1 - math.Exp(-avgSize/float64(rate)))

	return int64(float64(count) * scale), int64(float64(size) * scale)
}

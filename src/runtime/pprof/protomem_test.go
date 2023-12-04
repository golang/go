// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pprof

import (
	"bytes"
	"fmt"
	"internal/profile"
	"runtime"
	"slices"
	"strings"
	"testing"
)

func TestConvertMemProfile(t *testing.T) {
	addr1, addr2, map1, map2 := testPCs(t)

	// MemProfileRecord stacks are return PCs, so add one to the
	// addresses recorded in the "profile". The proto profile
	// locations are call PCs, so conversion will subtract one
	// from these and get back to addr1 and addr2.
	a1, a2 := uintptr(addr1)+1, uintptr(addr2)+1
	rate := int64(512 * 1024)
	rec := []runtime.MemProfileRecord{
		{AllocBytes: 4096, FreeBytes: 1024, AllocObjects: 4, FreeObjects: 1, Stack0: [32]uintptr{a1, a2}},
		{AllocBytes: 512 * 1024, FreeBytes: 0, AllocObjects: 1, FreeObjects: 0, Stack0: [32]uintptr{a2 + 1, a2 + 2}},
		{AllocBytes: 512 * 1024, FreeBytes: 512 * 1024, AllocObjects: 1, FreeObjects: 1, Stack0: [32]uintptr{a1 + 1, a1 + 2, a2 + 3}},
	}

	periodType := &profile.ValueType{Type: "space", Unit: "bytes"}
	sampleType := []*profile.ValueType{
		{Type: "alloc_objects", Unit: "count"},
		{Type: "alloc_space", Unit: "bytes"},
		{Type: "inuse_objects", Unit: "count"},
		{Type: "inuse_space", Unit: "bytes"},
	}
	samples := []*profile.Sample{
		{
			Value: []int64{2050, 2099200, 1537, 1574400},
			Location: []*profile.Location{
				{ID: 1, Mapping: map1, Address: addr1},
				{ID: 2, Mapping: map2, Address: addr2},
			},
			NumLabel: map[string][]int64{"bytes": {1024}},
		},
		{
			Value: []int64{1, 829411, 1, 829411},
			Location: []*profile.Location{
				{ID: 3, Mapping: map2, Address: addr2 + 1},
				{ID: 4, Mapping: map2, Address: addr2 + 2},
			},
			NumLabel: map[string][]int64{"bytes": {512 * 1024}},
		},
		{
			Value: []int64{1, 829411, 0, 0},
			Location: []*profile.Location{
				{ID: 5, Mapping: map1, Address: addr1 + 1},
				{ID: 6, Mapping: map1, Address: addr1 + 2},
				{ID: 7, Mapping: map2, Address: addr2 + 3},
			},
			NumLabel: map[string][]int64{"bytes": {512 * 1024}},
		},
	}
	for _, tc := range []struct {
		name              string
		defaultSampleType string
	}{
		{"heap", ""},
		{"allocs", "alloc_space"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var buf bytes.Buffer
			if err := writeHeapProto(&buf, rec, rate, tc.defaultSampleType); err != nil {
				t.Fatalf("writing profile: %v", err)
			}

			p, err := profile.Parse(&buf)
			if err != nil {
				t.Fatalf("profile.Parse: %v", err)
			}

			checkProfile(t, p, rate, periodType, sampleType, samples, tc.defaultSampleType)
		})
	}
}

func genericAllocFunc[T interface{ uint32 | uint64 }](n int) []T {
	return make([]T, n)
}

func profileToString(p *profile.Profile) []string {
	var res []string
	for _, s := range p.Sample {
		var funcs []string
		for i := len(s.Location) - 1; i >= 0; i-- {
			loc := s.Location[i]
			for j := len(loc.Line) - 1; j >= 0; j-- {
				line := loc.Line[j]
				funcs = append(funcs, line.Function.Name)
			}
		}
		res = append(res, fmt.Sprintf("%s %v", strings.Join(funcs, ";"), s.Value))
	}
	return res
}

// This is a regression test for https://go.dev/issue/64528 .
func TestGenericsHashKeyInPprofBuilder(t *testing.T) {
	previousRate := runtime.MemProfileRate
	runtime.MemProfileRate = 1
	defer func() {
		runtime.MemProfileRate = previousRate
	}()
	for _, sz := range []int{128, 256} {
		genericAllocFunc[uint32](sz / 4)
	}
	for _, sz := range []int{32, 64} {
		genericAllocFunc[uint64](sz / 8)
	}

	runtime.GC()
	buf := bytes.NewBuffer(nil)
	if err := WriteHeapProfile(buf); err != nil {
		t.Fatalf("writing profile: %v", err)
	}
	p, err := profile.Parse(buf)
	if err != nil {
		t.Fatalf("profile.Parse: %v", err)
	}

	actual := profileToString(p)
	expected := []string{
		"testing.tRunner;runtime/pprof.TestGenericsHashKeyInPprofBuilder;runtime/pprof.genericAllocFunc[go.shape.uint32] [1 128 0 0]",
		"testing.tRunner;runtime/pprof.TestGenericsHashKeyInPprofBuilder;runtime/pprof.genericAllocFunc[go.shape.uint32] [1 256 0 0]",
		"testing.tRunner;runtime/pprof.TestGenericsHashKeyInPprofBuilder;runtime/pprof.genericAllocFunc[go.shape.uint64] [1 32 0 0]",
		"testing.tRunner;runtime/pprof.TestGenericsHashKeyInPprofBuilder;runtime/pprof.genericAllocFunc[go.shape.uint64] [1 64 0 0]",
	}

	for _, l := range expected {
		if !slices.Contains(actual, l) {
			t.Errorf("profile = %v\nwant = %v", strings.Join(actual, "\n"), l)
		}
	}
}

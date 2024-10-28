// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pprof

import (
	"bytes"
	"fmt"
	"internal/asan"
	"internal/profile"
	"internal/profilerecord"
	"internal/testenv"
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
	rec := []profilerecord.MemProfileRecord{
		{AllocBytes: 4096, FreeBytes: 1024, AllocObjects: 4, FreeObjects: 1, Stack: []uintptr{a1, a2}},
		{AllocBytes: 512 * 1024, FreeBytes: 0, AllocObjects: 1, FreeObjects: 0, Stack: []uintptr{a2 + 1, a2 + 2}},
		{AllocBytes: 512 * 1024, FreeBytes: 512 * 1024, AllocObjects: 1, FreeObjects: 1, Stack: []uintptr{a1 + 1, a1 + 2, a2 + 3}},
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

func profileToStrings(p *profile.Profile) []string {
	var res []string
	for _, s := range p.Sample {
		res = append(res, sampleToString(s))
	}
	return res
}

func sampleToString(s *profile.Sample) string {
	var funcs []string
	for i := len(s.Location) - 1; i >= 0; i-- {
		loc := s.Location[i]
		funcs = locationToStrings(loc, funcs)
	}
	return fmt.Sprintf("%s %v", strings.Join(funcs, ";"), s.Value)
}

func locationToStrings(loc *profile.Location, funcs []string) []string {
	for j := range loc.Line {
		line := loc.Line[len(loc.Line)-1-j]
		funcs = append(funcs, line.Function.Name)
	}
	return funcs
}

// This is a regression test for https://go.dev/issue/64528 .
func TestGenericsHashKeyInPprofBuilder(t *testing.T) {
	if asan.Enabled {
		t.Skip("extra allocations with -asan throw off the test; see #70079")
	}
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

	actual := profileToStrings(p)
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

type opAlloc struct {
	buf [128]byte
}

type opCall struct {
}

var sink []byte

func storeAlloc() {
	sink = make([]byte, 16)
}

func nonRecursiveGenericAllocFunction[CurrentOp any, OtherOp any](alloc bool) {
	if alloc {
		storeAlloc()
	} else {
		nonRecursiveGenericAllocFunction[OtherOp, CurrentOp](true)
	}
}

func TestGenericsInlineLocations(t *testing.T) {
	if asan.Enabled {
		t.Skip("extra allocations with -asan throw off the test; see #70079")
	}
	if testenv.OptimizationOff() {
		t.Skip("skipping test with optimizations disabled")
	}

	previousRate := runtime.MemProfileRate
	runtime.MemProfileRate = 1
	defer func() {
		runtime.MemProfileRate = previousRate
		sink = nil
	}()

	nonRecursiveGenericAllocFunction[opAlloc, opCall](true)
	nonRecursiveGenericAllocFunction[opCall, opAlloc](false)

	runtime.GC()

	buf := bytes.NewBuffer(nil)
	if err := WriteHeapProfile(buf); err != nil {
		t.Fatalf("writing profile: %v", err)
	}
	p, err := profile.Parse(buf)
	if err != nil {
		t.Fatalf("profile.Parse: %v", err)
	}

	const expectedSample = "testing.tRunner;runtime/pprof.TestGenericsInlineLocations;runtime/pprof.nonRecursiveGenericAllocFunction[go.shape.struct {},go.shape.struct { runtime/pprof.buf [128]uint8 }];runtime/pprof.nonRecursiveGenericAllocFunction[go.shape.struct { runtime/pprof.buf [128]uint8 },go.shape.struct {}];runtime/pprof.storeAlloc [1 16 1 16]"
	const expectedLocation = "runtime/pprof.nonRecursiveGenericAllocFunction[go.shape.struct {},go.shape.struct { runtime/pprof.buf [128]uint8 }];runtime/pprof.nonRecursiveGenericAllocFunction[go.shape.struct { runtime/pprof.buf [128]uint8 },go.shape.struct {}];runtime/pprof.storeAlloc"
	const expectedLocationNewInliner = "runtime/pprof.TestGenericsInlineLocations;" + expectedLocation
	var s *profile.Sample
	for _, sample := range p.Sample {
		if sampleToString(sample) == expectedSample {
			s = sample
			break
		}
	}
	if s == nil {
		t.Fatalf("expected \n%s\ngot\n%s", expectedSample, strings.Join(profileToStrings(p), "\n"))
	}
	loc := s.Location[0]
	actual := strings.Join(locationToStrings(loc, nil), ";")
	if expectedLocation != actual && expectedLocationNewInliner != actual {
		t.Errorf("expected a location with at least 3 functions\n%s\ngot\n%s\n", expectedLocation, actual)
	}
}

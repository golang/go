// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package protopprof

import (
	"bytes"
	"internal/pprof/profile"
	"io/ioutil"
	"reflect"
	"runtime"
	"testing"
	"time"
)

// TestSampledHeapAllocProfile tests encoding of a memory profile from
// runtime.MemProfileRecord data.
func TestSampledHeapAllocProfile(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("Test requires a system with /proc/self/maps")
	}

	// Figure out two addresses from /proc/self/maps.
	mmap, err := ioutil.ReadFile("/proc/self/maps")
	if err != nil {
		t.Fatal("Cannot read /proc/self/maps")
	}
	rd := bytes.NewReader(mmap)
	mprof := &profile.Profile{}
	if err = mprof.ParseMemoryMap(rd); err != nil {
		t.Fatalf("Cannot parse /proc/self/maps")
	}
	if len(mprof.Mapping) < 2 {
		// It is possible for a binary to only have 1 executable
		// region of memory.
		t.Skipf("need 2 or more mappings, got %v", len(mprof.Mapping))
	}
	address1 := mprof.Mapping[0].Start
	address2 := mprof.Mapping[1].Start

	var buf bytes.Buffer

	rec, rate := testMemRecords(address1, address2)
	p := EncodeMemProfile(rec, rate, time.Now())
	if err := p.Write(&buf); err != nil {
		t.Fatalf("Failed to write profile: %v", err)
	}

	p, err = profile.Parse(&buf)
	if err != nil {
		t.Fatalf("Could not parse Profile profile: %v", err)
	}

	// Expected PeriodType, SampleType and Sample.
	expectedPeriodType := &profile.ValueType{Type: "space", Unit: "bytes"}
	expectedSampleType := []*profile.ValueType{
		{Type: "alloc_objects", Unit: "count"},
		{Type: "alloc_space", Unit: "bytes"},
		{Type: "inuse_objects", Unit: "count"},
		{Type: "inuse_space", Unit: "bytes"},
	}
	// Expected samples, with values unsampled according to the profiling rate.
	expectedSample := []*profile.Sample{
		{Value: []int64{2050, 2099200, 1537, 1574400}, Location: []*profile.Location{
			{ID: 1, Mapping: mprof.Mapping[0], Address: address1},
			{ID: 2, Mapping: mprof.Mapping[1], Address: address2},
		}},
		{Value: []int64{1, 829411, 1, 829411}, Location: []*profile.Location{
			{ID: 3, Mapping: mprof.Mapping[1], Address: address2 + 1},
			{ID: 4, Mapping: mprof.Mapping[1], Address: address2 + 2},
		}},
		{Value: []int64{1, 829411, 0, 0}, Location: []*profile.Location{
			{ID: 5, Mapping: mprof.Mapping[0], Address: address1 + 1},
			{ID: 6, Mapping: mprof.Mapping[0], Address: address1 + 2},
			{ID: 7, Mapping: mprof.Mapping[1], Address: address2 + 3},
		}},
	}

	if p.Period != 512*1024 {
		t.Fatalf("Sampling periods do not match")
	}
	if !reflect.DeepEqual(p.PeriodType, expectedPeriodType) {
		t.Fatalf("Period types do not match")
	}
	if !reflect.DeepEqual(p.SampleType, expectedSampleType) {
		t.Fatalf("Sample types do not match")
	}
	if !reflect.DeepEqual(p.Sample, expectedSample) {
		t.Fatalf("Samples do not match: Expected: %v, Got:%v", getSampleAsString(expectedSample),
			getSampleAsString(p.Sample))
	}
}

func testMemRecords(a1, a2 uint64) ([]runtime.MemProfileRecord, int64) {
	addr1, addr2 := uintptr(a1), uintptr(a2)
	rate := int64(512 * 1024)
	rec := []runtime.MemProfileRecord{
		{4096, 1024, 4, 1, [32]uintptr{addr1, addr2}},
		{512 * 1024, 0, 1, 0, [32]uintptr{addr2 + 1, addr2 + 2}},
		{512 * 1024, 512 * 1024, 1, 1, [32]uintptr{addr1 + 1, addr1 + 2, addr2 + 3}},
	}
	return rec, rate
}

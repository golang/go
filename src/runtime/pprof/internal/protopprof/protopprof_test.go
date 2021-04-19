// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package protopprof

import (
	"bytes"
	"fmt"
	"internal/pprof/profile"
	"io/ioutil"
	"reflect"
	"runtime"
	"testing"
	"time"
	"unsafe"
)

// Helper function to initialize empty cpu profile with sampling period provided.
func createEmptyProfileWithPeriod(t *testing.T, periodMs uint64) bytes.Buffer {
	// Mock the sample header produced by cpu profiler. Write a sample
	// period of 2000 microseconds, followed by no samples.
	buf := new(bytes.Buffer)
	// Profile header is as follows:
	// The first, third and fifth words are 0. The second word is 3.
	// The fourth word is the period.
	// EOD marker:
	// The sixth word -- count is initialized to 0 above.
	// The code below sets the seventh word -- nstk to 1
	// The eighth word -- addr is initialized to 0 above.
	words := []int{0, 3, 0, int(periodMs), 0, 0, 1, 0}
	n := int(unsafe.Sizeof(0)) * len(words)
	data := ((*[1 << 29]byte)(unsafe.Pointer(&words[0])))[:n:n]
	if _, err := buf.Write(data); err != nil {
		t.Fatalf("createEmptyProfileWithPeriod failed: %v", err)
	}
	return *buf
}

// Helper function to initialize cpu profile with two sample values.
func createProfileWithTwoSamples(t *testing.T, periodMs uintptr, count1 uintptr, count2 uintptr,
	address1 uintptr, address2 uintptr) bytes.Buffer {
	// Mock the sample header produced by cpu profiler. Write a sample
	// period of 2000 microseconds, followed by no samples.
	buf := new(bytes.Buffer)
	words := []uintptr{0, 3, 0, uintptr(periodMs), 0, uintptr(count1), 2,
		uintptr(address1), uintptr(address1 + 2),
		uintptr(count2), 2, uintptr(address2), uintptr(address2 + 2),
		0, 1, 0}
	for _, n := range words {
		var err error
		switch unsafe.Sizeof(int(0)) {
		case 8:
			_, err = buf.Write((*[8]byte)(unsafe.Pointer(&n))[:8:8])
		case 4:
			_, err = buf.Write((*[4]byte)(unsafe.Pointer(&n))[:4:4])
		}
		if err != nil {
			t.Fatalf("createProfileWithTwoSamples failed: %v", err)
		}
	}
	return *buf
}

// Tests TranslateCPUProfile parses correct sampling period in an otherwise empty cpu profile.
func TestTranlateCPUProfileSamplingPeriod(t *testing.T) {
	// A test server with mock cpu profile data.
	var buf bytes.Buffer

	startTime := time.Now()
	b := createEmptyProfileWithPeriod(t, 2000)
	p, err := TranslateCPUProfile(b.Bytes(), startTime)
	if err != nil {
		t.Fatalf("translate failed: %v", err)
	}
	if err := p.Write(&buf); err != nil {
		t.Fatalf("write failed: %v", err)
	}

	p, err = profile.Parse(&buf)
	if err != nil {
		t.Fatalf("Could not parse Profile profile: %v", err)
	}

	// Expected PeriodType and SampleType.
	expectedPeriodType := &profile.ValueType{Type: "cpu", Unit: "nanoseconds"}
	expectedSampleType := []*profile.ValueType{
		{Type: "samples", Unit: "count"},
		{Type: "cpu", Unit: "nanoseconds"},
	}
	if p.Period != 2000*1000 || !reflect.DeepEqual(p.PeriodType, expectedPeriodType) ||
		!reflect.DeepEqual(p.SampleType, expectedSampleType) || p.Sample != nil {
		t.Fatalf("Unexpected Profile fields")
	}
}

func getSampleAsString(sample []*profile.Sample) string {
	var str string
	for _, x := range sample {
		for _, y := range x.Location {
			if y.Mapping != nil {
				str += fmt.Sprintf("Mapping:%v\n", *y.Mapping)
			}
			str += fmt.Sprintf("Location:%v\n", y)
		}
		str += fmt.Sprintf("Sample:%v\n", *x)
	}
	return str
}

// Tests TranslateCPUProfile parses a cpu profile with sample values present.
func TestTranslateCPUProfileWithSamples(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("test requires a system with /proc/self/maps")
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
	// A test server with mock cpu profile data.

	startTime := time.Now()
	b := createProfileWithTwoSamples(t, 2000, 20, 40, uintptr(address1), uintptr(address2))
	p, err := TranslateCPUProfile(b.Bytes(), startTime)

	if err != nil {
		t.Fatalf("Could not parse Profile profile: %v", err)
	}
	// Expected PeriodType, SampleType and Sample.
	expectedPeriodType := &profile.ValueType{Type: "cpu", Unit: "nanoseconds"}
	expectedSampleType := []*profile.ValueType{
		{Type: "samples", Unit: "count"},
		{Type: "cpu", Unit: "nanoseconds"},
	}
	expectedSample := []*profile.Sample{
		{Value: []int64{20, 20 * 2000 * 1000}, Location: []*profile.Location{
			{ID: 1, Mapping: mprof.Mapping[0], Address: address1},
			{ID: 2, Mapping: mprof.Mapping[0], Address: address1 + 1},
		}},
		{Value: []int64{40, 40 * 2000 * 1000}, Location: []*profile.Location{
			{ID: 3, Mapping: mprof.Mapping[1], Address: address2},
			{ID: 4, Mapping: mprof.Mapping[1], Address: address2 + 1},
		}},
	}
	if p.Period != 2000*1000 {
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

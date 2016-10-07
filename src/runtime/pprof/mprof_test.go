// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pprof_test

import (
	"bytes"
	"math"
	"reflect"
	"runtime"
	. "runtime/pprof"
	"testing"
	"unsafe"
)

var memSink interface{}

func allocateTransient1M() {
	for i := 0; i < 1024; i++ {
		memSink = &struct{ x [1024]byte }{}
	}
}

//go:noinline
func allocateTransient2M() {
	memSink = make([]byte, 2<<20)
}

type Obj32 struct {
	link *Obj32
	pad  [32 - unsafe.Sizeof(uintptr(0))]byte
}

var persistentMemSink *Obj32

func allocatePersistent1K() {
	for i := 0; i < 32; i++ {
		// Can't use slice because that will introduce implicit allocations.
		obj := &Obj32{link: persistentMemSink}
		persistentMemSink = obj
	}
}

var memoryProfilerRun = 0

func TestMemoryProfiler(t *testing.T) {
	// Disable sampling, otherwise it's difficult to assert anything.
	oldRate := runtime.MemProfileRate
	runtime.MemProfileRate = 1
	defer func() {
		runtime.MemProfileRate = oldRate
	}()

	// Allocate a meg to ensure that mcache.next_sample is updated to 1.
	for i := 0; i < 1024; i++ {
		memSink = make([]byte, 1024)
	}

	// Do the interesting allocations.
	allocateTransient1M()
	allocateTransient2M()
	allocatePersistent1K()
	memSink = nil

	runtime.GC() // materialize stats
	var buf bytes.Buffer
	if err := Lookup("heap").WriteTo(&buf, 1); err != nil {
		t.Fatalf("failed to write heap profile: %v", err)
	}

	memoryProfilerRun++

	r := bytes.NewReader(buf.Bytes())
	p, err := Parse(r)
	if err != nil {
		t.Fatalf("can't parse pprof profile: %v", err)
	}
	if len(p.Sample) < 3 {
		t.Fatalf("few samples, got: %d", len(p.Sample))
	}
	testSample := make(map[int][]int64)
	testSample[0] = scaleHeapSample((int64)(32*memoryProfilerRun), (int64)(1024*memoryProfilerRun), p.Period)
	testSample[0] = append(testSample[0], testSample[0][0], testSample[0][1])
	testSample[1] = scaleHeapSample((int64)((1<<10)*memoryProfilerRun), (int64)((1<<20)*memoryProfilerRun), p.Period)
	testSample[1] = append([]int64{0, 0}, testSample[1][0], testSample[1][1])
	testSample[2] = scaleHeapSample((int64)(memoryProfilerRun), (int64)((2<<20)*memoryProfilerRun), p.Period)
	testSample[2] = append([]int64{0, 0}, testSample[2][0], testSample[2][1])
	for _, value := range testSample {
		found := false
		for i := range p.Sample {
			if reflect.DeepEqual(p.Sample[i].Value, value) {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("the entry did not match any sample:\n%v\n", value)
		}
	}
}

func scaleHeapSample(count, size, rate int64) []int64 {
	if count == 0 || size == 0 {
		return []int64{0, 0}
	}

	if rate <= 1 {
		// if rate==1 all samples were collected so no adjustment is needed.
		// if rate<1 treat as unknown and skip scaling.
		return []int64{count, size}
	}

	avgSize := float64(size) / float64(count)
	scale := 1 / (1 - math.Exp(-avgSize/float64(rate)))

	return []int64{int64(float64(count) * scale), int64(float64(size) * scale)}
}

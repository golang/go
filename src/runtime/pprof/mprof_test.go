// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !js

package pprof

import (
	"bytes"
	"fmt"
	"internal/profile"
	"reflect"
	"regexp"
	"runtime"
	"testing"
	"unsafe"
)

var memSink any

func allocateTransient1M() {
	for i := 0; i < 1024; i++ {
		memSink = &struct{ x [1024]byte }{}
	}
}

//go:noinline
func allocateTransient2M() {
	memSink = make([]byte, 2<<20)
}

func allocateTransient2MInline() {
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

// Allocate transient memory using reflect.Call.

func allocateReflectTransient() {
	memSink = make([]byte, 2<<20)
}

func allocateReflect() {
	rv := reflect.ValueOf(allocateReflectTransient)
	rv.Call(nil)
}

var memoryProfilerRun = 0

func TestMemoryProfiler(t *testing.T) {
	// Disable sampling, otherwise it's difficult to assert anything.
	oldRate := runtime.MemProfileRate
	runtime.MemProfileRate = 1
	defer func() {
		runtime.MemProfileRate = oldRate
	}()

	// Allocate a meg to ensure that mcache.nextSample is updated to 1.
	for i := 0; i < 1024; i++ {
		memSink = make([]byte, 1024)
	}

	// Do the interesting allocations.
	allocateTransient1M()
	allocateTransient2M()
	allocateTransient2MInline()
	allocatePersistent1K()
	allocateReflect()
	memSink = nil

	runtime.GC() // materialize stats

	memoryProfilerRun++

	tests := []struct {
		stk    []string
		legacy string
	}{{
		stk: []string{"runtime/pprof.allocatePersistent1K", "runtime/pprof.TestMemoryProfiler"},
		legacy: fmt.Sprintf(`%v: %v \[%v: %v\] @ 0x[0-9,a-f]+ 0x[0-9,a-f]+ 0x[0-9,a-f]+ 0x[0-9,a-f]+
#	0x[0-9,a-f]+	runtime/pprof\.allocatePersistent1K\+0x[0-9,a-f]+	.*/runtime/pprof/mprof_test\.go:47
#	0x[0-9,a-f]+	runtime/pprof\.TestMemoryProfiler\+0x[0-9,a-f]+	.*/runtime/pprof/mprof_test\.go:82
`, 32*memoryProfilerRun, 1024*memoryProfilerRun, 32*memoryProfilerRun, 1024*memoryProfilerRun),
	}, {
		stk: []string{"runtime/pprof.allocateTransient1M", "runtime/pprof.TestMemoryProfiler"},
		legacy: fmt.Sprintf(`0: 0 \[%v: %v\] @ 0x[0-9,a-f]+ 0x[0-9,a-f]+ 0x[0-9,a-f]+ 0x[0-9,a-f]+
#	0x[0-9,a-f]+	runtime/pprof\.allocateTransient1M\+0x[0-9,a-f]+	.*/runtime/pprof/mprof_test.go:24
#	0x[0-9,a-f]+	runtime/pprof\.TestMemoryProfiler\+0x[0-9,a-f]+	.*/runtime/pprof/mprof_test.go:79
`, (1<<10)*memoryProfilerRun, (1<<20)*memoryProfilerRun),
	}, {
		stk: []string{"runtime/pprof.allocateTransient2M", "runtime/pprof.TestMemoryProfiler"},
		legacy: fmt.Sprintf(`0: 0 \[%v: %v\] @ 0x[0-9,a-f]+ 0x[0-9,a-f]+ 0x[0-9,a-f]+ 0x[0-9,a-f]+
#	0x[0-9,a-f]+	runtime/pprof\.allocateTransient2M\+0x[0-9,a-f]+	.*/runtime/pprof/mprof_test.go:30
#	0x[0-9,a-f]+	runtime/pprof\.TestMemoryProfiler\+0x[0-9,a-f]+	.*/runtime/pprof/mprof_test.go:80
`, memoryProfilerRun, (2<<20)*memoryProfilerRun),
	}, {
		stk: []string{"runtime/pprof.allocateTransient2MInline", "runtime/pprof.TestMemoryProfiler"},
		legacy: fmt.Sprintf(`0: 0 \[%v: %v\] @ 0x[0-9,a-f]+ 0x[0-9,a-f]+ 0x[0-9,a-f]+ 0x[0-9,a-f]+
#	0x[0-9,a-f]+	runtime/pprof\.allocateTransient2MInline\+0x[0-9,a-f]+	.*/runtime/pprof/mprof_test.go:34
#	0x[0-9,a-f]+	runtime/pprof\.TestMemoryProfiler\+0x[0-9,a-f]+	.*/runtime/pprof/mprof_test.go:81
`, memoryProfilerRun, (2<<20)*memoryProfilerRun),
	}, {
		stk: []string{"runtime/pprof.allocateReflectTransient"},
		legacy: fmt.Sprintf(`0: 0 \[%v: %v\] @( 0x[0-9,a-f]+)+
#	0x[0-9,a-f]+	runtime/pprof\.allocateReflectTransient\+0x[0-9,a-f]+	.*/runtime/pprof/mprof_test.go:55
`, memoryProfilerRun, (2<<20)*memoryProfilerRun),
	}}

	t.Run("debug=1", func(t *testing.T) {
		var buf bytes.Buffer
		if err := Lookup("heap").WriteTo(&buf, 1); err != nil {
			t.Fatalf("failed to write heap profile: %v", err)
		}

		for _, test := range tests {
			if !regexp.MustCompile(test.legacy).Match(buf.Bytes()) {
				t.Fatalf("The entry did not match:\n%v\n\nProfile:\n%v\n", test.legacy, buf.String())
			}
		}
	})

	t.Run("proto", func(t *testing.T) {
		var buf bytes.Buffer
		if err := Lookup("heap").WriteTo(&buf, 0); err != nil {
			t.Fatalf("failed to write heap profile: %v", err)
		}
		p, err := profile.Parse(&buf)
		if err != nil {
			t.Fatalf("failed to parse heap profile: %v", err)
		}
		t.Logf("Profile = %v", p)

		stks := stacks(p)
		for _, test := range tests {
			if !containsStack(stks, test.stk) {
				t.Fatalf("No matching stack entry for %q\n\nProfile:\n%v\n", test.stk, p)
			}
		}

		if !containsInlinedCall(TestMemoryProfiler, 4<<10) {
			t.Logf("Can't determine whether allocateTransient2MInline was inlined into TestMemoryProfiler.")
			return
		}

		// Check the inlined function location is encoded correctly.
		for _, loc := range p.Location {
			inlinedCaller, inlinedCallee := false, false
			for _, line := range loc.Line {
				if line.Function.Name == "runtime/pprof.allocateTransient2MInline" {
					inlinedCallee = true
				}
				if inlinedCallee && line.Function.Name == "runtime/pprof.TestMemoryProfiler" {
					inlinedCaller = true
				}
			}
			if inlinedCallee != inlinedCaller {
				t.Errorf("want allocateTransient2MInline after TestMemoryProfiler in one location, got separate location entries:\n%v", loc)
			}
		}
	})
}

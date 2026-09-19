// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !js

package pprof

import (
	"bytes"
	"fmt"
	"internal/asan"
	"internal/profile"
	"reflect"
	"regexp"
	"runtime"
	"strconv"
	"strings"
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
	if asan.Enabled {
		t.Skip("extra allocations with -asan throw off the test; see #70079")
	}

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

	type entry struct {
		sizeEach   int
		liveCount  int
		allocCount int
	}

	tests := []struct {
		stk         []string
		legacy      string
		legacyEntry entry
	}{{
		// 4 PCs for the fast path
		// 5 PCs for the slow path with size-specialized malloc
		// 6 PCs for race builds (which also disable size-specialized malloc)
		stk: []string{"runtime/pprof.allocatePersistent1K", "runtime/pprof.TestMemoryProfiler"},
		legacy: `([0-9]+): ([0-9]+) \[([0-9]+): ([0-9]+)\] @( 0x[0-9,a-f]+){4,6}
#	0x[0-9,a-f]+	runtime/pprof\.allocatePersistent1K\+0x[0-9,a-f]+	.*runtime/pprof/mprof_test\.go:50
#	0x[0-9,a-f]+	runtime/pprof\.TestMemoryProfiler\+0x[0-9,a-f]+	.*runtime/pprof/mprof_test\.go:89
`,
		legacyEntry: entry{sizeEach: 32, liveCount: 32, allocCount: 32},
	}, {
		stk: []string{"runtime/pprof.allocateTransient1M", "runtime/pprof.TestMemoryProfiler"},
		legacy: `([0-9]+): ([0-9]+) \[([0-9]+): ([0-9]+)\] @ 0x[0-9,a-f]+ 0x[0-9,a-f]+ 0x[0-9,a-f]+ 0x[0-9,a-f]+ 0x[0-9,a-f]+ 0x[0-9,a-f]+
#	0x[0-9,a-f]+	runtime/pprof\.allocateTransient1M\+0x[0-9,a-f]+	.*runtime/pprof/mprof_test.go:27
#	0x[0-9,a-f]+	runtime/pprof\.TestMemoryProfiler\+0x[0-9,a-f]+	.*runtime/pprof/mprof_test.go:86
`,
		legacyEntry: entry{sizeEach: 1 << 10, allocCount: 1 << 10},
	}, {
		stk: []string{"runtime/pprof.allocateTransient2M", "runtime/pprof.TestMemoryProfiler"},
		legacy: `([0-9]+): ([0-9]+) \[([0-9]+): ([0-9]+)\] @ 0x[0-9,a-f]+ 0x[0-9,a-f]+ 0x[0-9,a-f]+ 0x[0-9,a-f]+ 0x[0-9,a-f]+ 0x[0-9,a-f]+
#	0x[0-9,a-f]+	runtime/pprof\.allocateTransient2M\+0x[0-9,a-f]+	.*runtime/pprof/mprof_test.go:33
#	0x[0-9,a-f]+	runtime/pprof\.TestMemoryProfiler\+0x[0-9,a-f]+	.*runtime/pprof/mprof_test.go:87
`,
		legacyEntry: entry{sizeEach: 2 << 20, allocCount: 1},
	}, {
		stk: []string{"runtime/pprof.allocateTransient2MInline", "runtime/pprof.TestMemoryProfiler"},
		legacy: `([0-9]+): ([0-9]+) \[([0-9]+): ([0-9]+)\] @ 0x[0-9,a-f]+ 0x[0-9,a-f]+ 0x[0-9,a-f]+ 0x[0-9,a-f]+ 0x[0-9,a-f]+ 0x[0-9,a-f]+
#	0x[0-9,a-f]+	runtime/pprof\.allocateTransient2MInline\+0x[0-9,a-f]+	.*runtime/pprof/mprof_test.go:37
#	0x[0-9,a-f]+	runtime/pprof\.TestMemoryProfiler\+0x[0-9,a-f]+	.*runtime/pprof/mprof_test.go:88
`,
		legacyEntry: entry{sizeEach: 2 << 20, allocCount: 1},
	}, {
		stk: []string{"runtime/pprof.allocateReflectTransient"},
		legacy: `([0-9]+): ([0-9]+) \[([0-9]+): ([0-9]+)\] @( 0x[0-9,a-f]+)+
#	0x[0-9,a-f]+	runtime/pprof\.allocateReflectTransient\+0x[0-9,a-f]+	.*runtime/pprof/mprof_test.go:58
`,
		legacyEntry: entry{sizeEach: 2 << 20, allocCount: 1},
	}}

	t.Run("debug=1", func(t *testing.T) {
		var buf bytes.Buffer
		if err := Lookup("heap").WriteTo(&buf, 1); err != nil {
			t.Fatalf("failed to write heap profile: %v", err)
		}

		defer func() {
			if t.Failed() {
				t.Logf("\nProfile:\n%v\n", buf.String())
			}
		}()

		for _, test := range tests {
			re := regexp.MustCompile(test.legacy)
			// Small allocations can appear with more than one call stack, such
			// as the fast vs slow paths in the size-specialized malloc code.
			// Individual line numbers can also be responsible for multiple
			// allocations. That includes not only the byte slices that these
			// tests try to observe, but also the slice headers (easy to split
			// out) as well as runtime-internal structures such as sudogs
			// (harder)! Filter by the size of the entry to see if it's the one
			// we're trying to observe, and sum all entries of that size.

			var (
				wantSize   = test.legacyEntry.sizeEach
				wantLive   = memoryProfilerRun * test.legacyEntry.liveCount
				wantAllocs = memoryProfilerRun * test.legacyEntry.allocCount

				foundLive   int
				foundAllocs int
			)

			var matches []string
			for _, match := range re.FindAllSubmatch(buf.Bytes(), -1) {
				if len(match) < 5 {
					continue
				}
				matches = append(matches, string(match[0]))
				liveCount, _ := strconv.Atoi(string(match[1]))
				allocCount, _ := strconv.Atoi(string(match[3]))
				totalSize, _ := strconv.Atoi(string(match[4]))
				if allocCount == 0 {
					continue
				}
				sizeEach := totalSize / allocCount
				if sizeEach == wantSize {
					foundLive += liveCount
					foundAllocs += allocCount
				}
			}

			if foundLive != wantLive || foundAllocs != wantAllocs {
				t.Errorf("Found %d entries with value %v (not %v) matching\n%v\n\n%v\n", len(matches),
					fmt.Sprintf("%d: %d [%d: %d]", foundLive, foundLive*wantSize, foundAllocs, foundAllocs*wantSize),
					fmt.Sprintf("%d: %d [%d: %d]", wantLive, wantLive*wantSize, wantAllocs, wantAllocs*wantSize),
					test.legacy, strings.Join(matches, "\n\n"))
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

		stks := profileStacks(p)
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

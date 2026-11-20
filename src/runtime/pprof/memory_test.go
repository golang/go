// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pprof

import (
	"bytes"
	"internal/profile"
	"internal/profilerecord"
	"internal/testenv"
	"io"
	"maps"
	"os"
	"runtime"
	"runtime/debug"
	"slices"
	"strings"
	"testing"
)

func TestMemoryProfileRSSUnsupported(t *testing.T) {
	if runtime.GOOS == "linux" || runtime.GOOS == "darwin" {
		return
	}
	var buf bytes.Buffer
	if err := Lookup("memory").WriteTo(&buf, 0); err != nil {
		t.Fatalf("writing memory profile: %v", err)
	}
	p, err := profile.Parse(&buf)
	if err != nil {
		t.Fatalf("parsing memory profile: %v", err)
	}
	if !slices.Contains(p.Comments, "RSS is unavailable on this system") {
		t.Error("memory profile does not report unavailable RSS")
	}
}

func TestMemoryProfileDisabled(t *testing.T) {
	if os.Getenv("GO_TEST_MEMORYPROFILE_DISABLED") == "" {
		// GODEBUG=memoryprofile is read at startup, so re-run this test
		// in a subprocess with the setting applied.
		cmd := testenv.CleanCmdEnv(testenv.Command(t, testenv.Executable(t),
			"-test.run=^"+t.Name()+"$", "-test.v"))
		cmd.Env = append(cmd.Env, "GO_TEST_MEMORYPROFILE_DISABLED=1", "GODEBUG=memoryprofile=0")
		out, err := cmd.CombinedOutput()
		if err != nil || !bytes.Contains(out, []byte("--- PASS: "+t.Name())) {
			t.Fatalf("subprocess failed: %v\n%s", err, out)
		}
		return
	}

	err := Lookup("memory").WriteTo(io.Discard, 0)
	if err == nil || !strings.Contains(err.Error(), "memoryprofile=0") {
		t.Errorf("writing memory profile: got error %v, want error mentioning memoryprofile=0", err)
	}

	// The heap profile does not depend on the memory profile metrics.
	var buf bytes.Buffer
	if err := Lookup("heap").WriteTo(&buf, 0); err != nil {
		t.Fatalf("writing heap profile: %v", err)
	}
	if _, err := profile.Parse(&buf); err != nil {
		t.Fatalf("parsing heap profile: %v", err)
	}
}

func TestMemoryProfileStructure(t *testing.T) {
	// Do a GC so we have heap metrics
	runtime.GC()
	var buf bytes.Buffer
	if err := Lookup("memory").WriteTo(&buf, 0); err != nil {
		t.Fatalf("writing memory profile: %v", err)
	}

	p, err := profile.Parse(&buf)
	if err != nil {
		t.Fatalf("parsing memory profile: %v", err)
	}
	t.Logf("%v", p)
	if len(p.SampleType) != 1 {
		t.Fatalf("got %d sample types, want 1", len(p.SampleType))
	}
	if st := p.SampleType[0]; st.Type != "memory_space" || st.Unit != "bytes" {
		t.Fatalf("sample type = {%q, %q}, want {memory_space, bytes}", st.Type, st.Unit)
	}
	if pt := p.PeriodType; pt.Type != "space" || pt.Unit != "bytes" {
		t.Fatalf("period type = {%q, %q}, want {space, bytes}", pt.Type, pt.Unit)
	}
	if len(p.Sample) == 0 {
		// We should see at least some of the runtime memory metrics in every profile
		t.Fatal("empty profile")
	}

	// Every traceback in the profile should have one of these prefixes
	virtualPrefixes := []string{
		rssPath(rssFrameGoMemory, rssFrameStacks, rssFrameGoroutineStacks),
		rssPath(rssFrameGoMemory, rssFrameStacks, rssFrameOSStacks),
		rssPath(rssFrameGoMemory, rssFrameRuntime, rssFrameMetadata, rssFrameMSpan, rssFrameInUseMSpans),
		rssPath(rssFrameGoMemory, rssFrameRuntime, rssFrameMetadata, rssFrameMSpan, rssFrameFreeMSpans),
		rssPath(rssFrameGoMemory, rssFrameRuntime, rssFrameMetadata, rssFrameMCache, rssFrameInUseMCaches),
		rssPath(rssFrameGoMemory, rssFrameRuntime, rssFrameMetadata, rssFrameMCache, rssFrameFreeMCaches),
		rssPath(rssFrameGoMemory, rssFrameRuntime, rssFrameMetadata, rssFrameOtherMetadata),
		rssPath(rssFrameGoMemory, rssFrameRuntime, rssFrameProfilingBuckets),
		rssPath(rssFrameGoMemory, rssFrameRuntime, rssFrameOtherRuntime),
		rssPath(rssFrameGoMemory, rssFrameHeap, rssFrameLiveHeap),
		rssPath(rssFrameGoMemory, rssFrameHeap, rssFrameDeadHeap),
		rssPath(rssFrameGoMemory, rssFrameHeap, rssFrameUnusedHeap),
		rssPath(rssFrameGoMemory, rssFrameHeap, rssFrameFreeHeap),
	}
	if runtime.GOOS == "linux" {
		virtualPrefixes = append(virtualPrefixes,
			rssPath(rssFrameNonGoMemory, rssFrameFileShared),
			rssPath(rssFrameNonGoMemory, rssFrameAnonymous))
	} else {
		virtualPrefixes = append(virtualPrefixes, rssPath(rssFrameNonGoMemory))
	}

	livePrefix := rssPath(rssFrameGoMemory, rssFrameHeap, rssFrameLiveHeap)
	deadPrefix := rssPath(rssFrameGoMemory, rssFrameHeap, rssFrameDeadHeap)
	seenPrefixes := make(map[string]bool)
	var goMemory, heap int64
	for _, s := range p.Sample {
		if len(s.Value) != 1 {
			t.Fatalf("sample has %d values, want 1", len(s.Value))
		}
		if s.Value[0] < 0 {
			t.Errorf("negative sample value: %d", s.Value[0])
		}
		var frames []string
		for _, loc := range slices.Backward(s.Location) {
			for _, line := range slices.Backward(loc.Line) {
				frames = append(frames, line.Function.Name)
			}
		}
		traceback := strings.Join(frames, ";")
		var prefix string
		for _, p := range virtualPrefixes {
			if strings.HasPrefix(traceback, p) {
				prefix = p
				break
			}
		}
		if prefix == "" {
			t.Errorf("memory profile traceback %q does not start with an expected virtual memory prefix", traceback)
			continue
		}
		seenPrefixes[prefix] = true
		for _, p := range []string{livePrefix, deadPrefix} {
			if !strings.HasPrefix(traceback, p) {
				continue
			}
			if traceback != p {
				// If there is a traceback following the prefix, the sample
				// should have an object size label
				sizes := s.NumLabel["bytes"]
				if len(sizes) != 1 || sizes[0] <= 0 {
					t.Errorf("%s sample has bytes size labels %v, want one positive size", p, sizes)
				}
			}
			break
		}
		if strings.HasPrefix(prefix, rssPath(rssFrameGoMemory)) {
			goMemory += s.Value[0]
		}
		if strings.HasPrefix(prefix, rssPath(rssFrameGoMemory, rssFrameHeap)) {
			heap += s.Value[0]
		}
	}

	if !seenPrefixes[livePrefix] {
		t.Errorf("profile has no %s sample", livePrefix)
	}
	if goMemory <= 0 {
		t.Errorf("sum of Go-memory samples = %d, want positive", goMemory)
	}
	if heap <= 0 {
		t.Errorf("sum of heap samples = %d, want positive", heap)
	}
	if heap > goMemory {
		t.Errorf("sum of heap samples = %d, exceeds Go-memory samples = %d", heap, goMemory)
	}
}

const rssProfileCycleAllocationSize = 50 << 20

var rssProfileCycleSink []byte

//go:noinline
func allocateRSSProfileLive() []byte {
	return make([]byte, rssProfileCycleAllocationSize)
}

//go:noinline
func allocateRSSProfileStale() {
	rssProfileCycleSink = make([]byte, rssProfileCycleAllocationSize)
}

//go:noinline
func allocateRSSProfileLatest() {
	rssProfileCycleSink = make([]byte, rssProfileCycleAllocationSize)
}

func TestMemoryProfileHeapSamples(t *testing.T) {
	// Disable automatic GC so reading a profile cannot change which GC it describes.
	oldGCPercent := debug.SetGCPercent(-1)
	defer debug.SetGCPercent(oldGCPercent)

	oldRate := runtime.MemProfileRate
	runtime.MemProfileRate = 1
	defer func() { runtime.MemProfileRate = oldRate }()

	// Keep a live allocation through the profile snapshot.
	live := allocateRSSProfileLive()
	runtime.GC()

	// The memory profile should show the state of the heap as of the latest
	// complete garbage collection, so we shouldn't see these.
	for range 3 {
		allocateRSSProfileStale()
		rssProfileCycleSink = nil
		runtime.GC()
	}

	allocateRSSProfileLatest()
	rssProfileCycleSink = nil
	runtime.GC()

	const liveName = "runtime/pprof.allocateRSSProfileLive"
	const stale = "runtime/pprof.allocateRSSProfileStale"
	const latest = "runtime/pprof.allocateRSSProfileLatest"
	checkProfile := func(wantDead bool) {
		t.Helper()
		var buf bytes.Buffer
		if err := Lookup("memory").WriteTo(&buf, 0); err != nil {
			t.Fatalf("writing memory profile: %v", err)
		}
		p, err := profile.Parse(&buf)
		if err != nil {
			t.Fatalf("parsing memory profile: %v", err)
		}
		t.Logf("%v", p)

		var liveSpace, staleDead, latestDead int64
		for _, sample := range p.Sample {
			if rssSampleHasFrame(sample, string(rssFrameLiveHeap)) && rssSampleHasFrame(sample, liveName) {
				liveSpace += sample.Value[0]
			}
			if rssSampleHasFrame(sample, string(rssFrameDeadHeap)) {
				if rssSampleHasFrame(sample, stale) {
					staleDead += sample.Value[0]
				}
				if rssSampleHasFrame(sample, latest) {
					latestDead += sample.Value[0]
				}
			}
		}
		if liveSpace == 0 {
			t.Error("no [Live Heap] space attributed to allocateRSSProfileLive")
		}
		if staleDead != 0 {
			t.Errorf("[Dead Heap] space attributed to allocations freed in earlier GC cycles = %d, want 0", staleDead)
		}
		if wantDead && latestDead == 0 {
			t.Error("no [Dead Heap] space attributed to allocation freed in the final GC cycle")
		} else if !wantDead && latestDead != 0 {
			t.Errorf("[Dead Heap] space retained from an earlier GC = %d, want 0", latestDead)
		}
	}

	// The profile should show memory freed by the last GC.
	checkProfile(true)

	// Reading the profile again should leave those counts unchanged.
	checkProfile(true)

	// The profiler rotates through three sets of counters. Run three more
	// GCs to return to the set that held the latest frees. We don't allocate
	// at that site again, so those frees should disappear after the first GC
	// and must not reappear when the profile reads the same counters again.
	for range 3 {
		runtime.GC()
		checkProfile(false)
	}

	runtime.KeepAlive(live)
}

func TestMemoryProfileRuntimeLeafFrames(t *testing.T) {
	oldGCPercent := debug.SetGCPercent(-1)
	defer debug.SetGCPercent(oldGCPercent)

	previousRate := runtime.MemProfileRate
	runtime.MemProfileRate = 1
	defer func() {
		runtime.MemProfileRate = previousRate
	}()

	growMap()

	runtime.GC()
	var buf bytes.Buffer
	if err := Lookup("memory").WriteTo(&buf, 0); err != nil {
		t.Fatalf("writing profile: %v", err)
	}
	p, err := profile.Parse(&buf)
	if err != nil {
		t.Fatalf("profile.Parse: %v", err)
	}

	actual := profileToStrings(p)

	// We must see growMap at least once.
	foundGrowMap := false
	for _, l := range actual {
		if !strings.Contains(l, "runtime/pprof.growMap") {
			continue
		}
		foundGrowMap = true

		// Runtime frames like mapassign and map internals should be hidden.
		if strings.Contains(l, "runtime.") {
			t.Errorf("Sample got %s, want no runtime frames", l)
		}
		if strings.Contains(l, "internal/runtime/") {
			t.Errorf("Sample got %s, want no runtime frames", l)
		}
		if strings.Contains(l, "mapassign") { // in case mapassign moves to a package not matching above paths.
			t.Errorf("Sample got %s, want no mapassign frames", l)
		}
	}

	if !foundGrowMap {
		t.Errorf("Profile got:\n%s\nwant sample in runtime/pprof.growMap", strings.Join(actual, "\n"))
	}
}

func rssSampleHasFrame(sample *profile.Sample, name string) bool {
	for _, loc := range sample.Location {
		for _, line := range loc.Line {
			if line.Function.Name == name {
				return true
			}
		}
	}
	return false
}

func TestMemoryProfileNonGoBreakdown(t *testing.T) {
	// Go memory is 100 bytes: 60 bytes of heap objects (40 live, 20 dead)
	// and 40 bytes of other runtime memory.
	goMetrics := profilerecord.MemProfileMetrics{
		HeapLive:                 40,
		MemoryClassesHeapObjects: 60,
		MemoryClassesOther:       40,
		MemoryClassesTotal:       100,
	}
	fileShared := rssPath(rssFrameNonGoMemory, rssFrameFileShared)
	anonymous := rssPath(rssFrameNonGoMemory, rssFrameAnonymous)
	nonGo := rssPath(rssFrameNonGoMemory)

	for _, tc := range []struct {
		name        string
		rss         uint64
		shared      uint64
		hasShared   bool
		want        map[string]int64 // non-Go traceback -> bytes
		wantComment string
	}{
		{
			name: "split", rss: 150, shared: 30, hasShared: true,
			want: map[string]int64{fileShared: 30, anonymous: 20},
		},
		{
			name: "anonymous below Go memory", rss: 150, shared: 70, hasShared: true,
			want:        map[string]int64{fileShared: 70},
			wantComment: "Some Go Memory is not resident: Go Memory (100 bytes) > Anonymous RSS (80 bytes). Anonymous Non-Go Memory not shown.",
		},
		{
			name: "RSS below Go memory", rss: 90, shared: 10, hasShared: true,
			want:        map[string]int64{fileShared: 10},
			wantComment: "Some Go Memory is not resident: Go Memory (100 bytes) > Anonymous RSS (80 bytes). Anonymous Non-Go Memory not shown.",
		},
		{
			name: "anonymous equals Go memory", rss: 150, shared: 50, hasShared: true,
			want: map[string]int64{fileShared: 50},
		},
		{
			name: "no breakdown", rss: 150,
			want: map[string]int64{nonGo: 50},
		},
		{
			name: "no breakdown, RSS equals Go memory", rss: 100,
			want: map[string]int64{},
		},
		{
			name: "no breakdown, RSS below Go memory", rss: 90,
			want:        map[string]int64{},
			wantComment: "Some Go Memory is not resident: Go Memory (100 bytes) > RSS (90 bytes). Non-Go Memory not shown.",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := goMetrics
			m.RSS, m.RSSShared, m.RSSHasShared = tc.rss, tc.shared, tc.hasShared
			var buf bytes.Buffer
			if err := writeMemoryProto(&buf, nil, 1, &m); err != nil {
				t.Fatalf("writing memory profile: %v", err)
			}
			p, err := profile.Parse(&buf)
			if err != nil {
				t.Fatalf("parsing memory profile: %v", err)
			}
			got := make(map[string]int64)
			for _, s := range p.Sample {
				var frames []string
				for _, loc := range slices.Backward(s.Location) {
					frames = append(frames, loc.Line[0].Function.Name)
				}
				if traceback := strings.Join(frames, ";"); strings.HasPrefix(traceback, nonGo) {
					got[traceback] += s.Value[0]
				}
			}
			if !maps.Equal(got, tc.want) {
				t.Errorf("non-Go samples = %v, want %v", got, tc.want)
			}
			if tc.wantComment == "" && len(p.Comments) != 0 {
				t.Errorf("comments = %q, want none", p.Comments)
			}
			if tc.wantComment != "" && !slices.Contains(p.Comments, tc.wantComment) {
				t.Errorf("comments = %q, want %q", p.Comments, tc.wantComment)
			}
		})
	}
}

func rssPath(frames ...rssVirtualFrame) string {
	parts := make([]string, len(frames))
	for i, f := range frames {
		parts[i] = string(f)
	}
	return strings.Join(parts, ";")
}

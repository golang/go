// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pprof

import (
	"errors"
	"fmt"
	"internal/profilerecord"
	"io"
	"math/bits"
	"runtime"
	"slices"
)

// rssVirtualFrame is a virtual frame name used in the memory profile to represent
// memory categories. The bracketed names cannot collide with real Go function names.
type rssVirtualFrame string

const (
	rssFrameGoMemory         rssVirtualFrame = "[Go Memory]"
	rssFrameNonGoMemory      rssVirtualFrame = "[Non-Go Memory]"
	rssFrameStacks           rssVirtualFrame = "[Stacks]"
	rssFrameGoroutineStacks  rssVirtualFrame = "[Goroutine Stacks]"
	rssFrameOSStacks         rssVirtualFrame = "[OS Stacks]"
	rssFrameRuntime          rssVirtualFrame = "[Runtime]"
	rssFrameMetadata         rssVirtualFrame = "[Metadata]"
	rssFrameMSpan            rssVirtualFrame = "[MSpan]"
	rssFrameInUseMSpans      rssVirtualFrame = "[In-Use MSpans]"
	rssFrameFreeMSpans       rssVirtualFrame = "[Free MSpans]"
	rssFrameMCache           rssVirtualFrame = "[MCache]"
	rssFrameInUseMCaches     rssVirtualFrame = "[In-Use MCaches]"
	rssFrameFreeMCaches      rssVirtualFrame = "[Free MCaches]"
	rssFrameOtherMetadata    rssVirtualFrame = "[Other Metadata]"
	rssFrameProfilingBuckets rssVirtualFrame = "[Profiling Buckets]"
	rssFrameOtherRuntime     rssVirtualFrame = "[Other Runtime]"
	rssFrameHeap             rssVirtualFrame = "[Heap]"
	rssFrameLiveHeap         rssVirtualFrame = "[Live Heap]"
	rssFrameDeadHeap         rssVirtualFrame = "[Dead Heap]"
	rssFrameUnusedHeap       rssVirtualFrame = "[Unused Heap]"
	rssFrameFreeHeap         rssVirtualFrame = "[Free Heap]"
)

var memoryProfile = &Profile{
	name:  "memory",
	count: countMemory,
	write: writeMemory,
}

func countMemory() int { return 0 }

func writeMemory(w io.Writer, debug int) error {
	if debug != 0 {
		return fmt.Errorf("memory profile: only debug=0 is supported")
	}
	if !pprof_memoryProfileEnabled() {
		return errors.New("memory profile: disabled by GODEBUG=memoryprofile=0")
	}
	p, metrics := readMemProfileWithMetrics()
	if err := validateRSSMetrics(&metrics); err != nil {
		return fmt.Errorf("memory profile: %w", err)
	}
	return writeMemoryProto(w, p, int64(runtime.MemProfileRate), &metrics)
}

func readMemProfileWithMetrics() ([]profilerecord.MemProfileRecord, profilerecord.MemProfileMetrics) {
	var p []profilerecord.MemProfileRecord
	var metrics profilerecord.MemProfileMetrics
	n, ok := pprof_memProfileInternal(nil, true, nil)
	for {
		p = make([]profilerecord.MemProfileRecord, n+50)
		n, ok = pprof_memProfileInternal(p, true, &metrics)
		if ok {
			return p[:n], metrics
		}
	}
}

func validateRSSMetrics(m *profilerecord.MemProfileMetrics) error {
	leafSum := m.MemoryClassesHeapStacks + m.MemoryClassesOSStacks +
		m.MemoryClassesMetadataMSpanInuse + m.MemoryClassesMetadataMSpanFree +
		m.MemoryClassesMetadataMCacheInuse + m.MemoryClassesMetadataMCacheFree +
		m.MemoryClassesMetadataOther + m.MemoryClassesProfilingBuckets + m.MemoryClassesOther +
		m.MemoryClassesHeapFree + m.MemoryClassesHeapObjects + m.MemoryClassesHeapUnused +
		m.MemoryClassesHeapReleased
	if leafSum != m.MemoryClassesTotal {
		return fmt.Errorf("memory/classes leaf sum (%d) != total (%d)", leafSum, m.MemoryClassesTotal)
	}
	heapTotal := m.MemoryClassesHeapFree + m.MemoryClassesHeapObjects + m.MemoryClassesHeapUnused
	if m.HeapLive > heapTotal {
		return fmt.Errorf("heap live (%d) > heap total (%d)", m.HeapLive, heapTotal)
	}
	return nil
}

func writeMemoryProto(w io.Writer, p []profilerecord.MemProfileRecord, rate int64, m *profilerecord.MemProfileMetrics) error {
	b := newProfileBuilder(w)
	b.pbValueType(tagProfile_PeriodType, "space", "bytes")
	b.pb.int64Opt(tagProfile_Period, 1)
	b.pbValueType(tagProfile_SampleType, "memory_space", "bytes")

	type heapWeights struct {
		live, dead int64
	}
	weights := make([]heapWeights, len(p))
	var liveWeightTotal, deadWeightTotal int64
	for i, r := range p {
		_, weights[i].live = scaleHeapSample(r.InUseObjects(), r.ObjectSize, rate)
		_, weights[i].dead = scaleHeapSample(r.FreedLastGC, r.ObjectSize, rate)
		if weights[i].live > 0 {
			liveWeightTotal += weights[i].live
		}
		if weights[i].dead > 0 {
			deadWeightTotal += weights[i].dead
		}
	}

	frames := []rssVirtualFrame{
		rssFrameGoMemory, rssFrameNonGoMemory,
		rssFrameStacks, rssFrameGoroutineStacks, rssFrameOSStacks,
		rssFrameRuntime, rssFrameMetadata,
		rssFrameMSpan, rssFrameInUseMSpans, rssFrameFreeMSpans,
		rssFrameMCache, rssFrameInUseMCaches, rssFrameFreeMCaches,
		rssFrameOtherMetadata, rssFrameProfilingBuckets, rssFrameOtherRuntime,
		rssFrameHeap, rssFrameLiveHeap, rssFrameDeadHeap, rssFrameUnusedHeap, rssFrameFreeHeap,
	}
	vLoc := make(map[rssVirtualFrame]uint64, len(frames))
	for _, frame := range frames {
		name := string(frame)
		funcID := b.reserveFuncID()
		fStart := b.pb.startMessage()
		b.pb.uint64Opt(tagFunction_ID, funcID)
		b.pb.int64Opt(tagFunction_Name, b.stringIndex(name))
		b.pb.int64Opt(tagFunction_SystemName, b.stringIndex(name))
		b.pb.int64Opt(tagFunction_Filename, b.stringIndex("[memory profile]"))
		b.pb.endMessage(tagProfile_Function, fStart)
		locID := b.reserveLocID()
		lStart := b.pb.startMessage()
		b.pb.uint64Opt(tagLocation_ID, locID)
		b.pbLine(tagLocation_Line, funcID, 0)
		b.pb.endMessage(tagProfile_Location, lStart)
		vLoc[frame] = locID
	}

	var sampleLocs []uint64
	emitSample := func(value int64, vpath []rssVirtualFrame, realLocs []uint64, size int64) {
		if value == 0 {
			return
		}
		sampleLocs = append(sampleLocs[:0], realLocs...)
		for _, v := range slices.Backward(vpath) {
			sampleLocs = append(sampleLocs, vLoc[v])
		}
		var labels func()
		if size > 0 {
			labels = func() { b.pbLabel(tagSample_Label, "bytes", "", size) }
		}
		// pbSample encodes the locations before returning, so the slice can be reused.
		b.pbSample([]int64{value}, sampleLocs, labels)
	}

	fixed := []struct {
		value uint64
		path  []rssVirtualFrame
	}{
		{m.MemoryClassesHeapStacks, []rssVirtualFrame{rssFrameGoMemory, rssFrameStacks, rssFrameGoroutineStacks}},
		{m.MemoryClassesOSStacks, []rssVirtualFrame{rssFrameGoMemory, rssFrameStacks, rssFrameOSStacks}},
		{m.MemoryClassesMetadataMSpanInuse, []rssVirtualFrame{rssFrameGoMemory, rssFrameRuntime, rssFrameMetadata, rssFrameMSpan, rssFrameInUseMSpans}},
		{m.MemoryClassesMetadataMSpanFree, []rssVirtualFrame{rssFrameGoMemory, rssFrameRuntime, rssFrameMetadata, rssFrameMSpan, rssFrameFreeMSpans}},
		{m.MemoryClassesMetadataMCacheInuse, []rssVirtualFrame{rssFrameGoMemory, rssFrameRuntime, rssFrameMetadata, rssFrameMCache, rssFrameInUseMCaches}},
		{m.MemoryClassesMetadataMCacheFree, []rssVirtualFrame{rssFrameGoMemory, rssFrameRuntime, rssFrameMetadata, rssFrameMCache, rssFrameFreeMCaches}},
		{m.MemoryClassesMetadataOther, []rssVirtualFrame{rssFrameGoMemory, rssFrameRuntime, rssFrameMetadata, rssFrameOtherMetadata}},
		{m.MemoryClassesProfilingBuckets, []rssVirtualFrame{rssFrameGoMemory, rssFrameRuntime, rssFrameProfilingBuckets}},
		{m.MemoryClassesOther, []rssVirtualFrame{rssFrameGoMemory, rssFrameRuntime, rssFrameOtherRuntime}},
	}
	for _, f := range fixed {
		emitSample(int64(f.value), f.path, nil, 0)
	}

	livePath := []rssVirtualFrame{rssFrameGoMemory, rssFrameHeap, rssFrameLiveHeap}
	deadPath := []rssVirtualFrame{rssFrameGoMemory, rssFrameHeap, rssFrameDeadHeap}
	goMemory := m.MemoryClassesTotal - m.MemoryClassesHeapReleased
	heapLive := int64(m.HeapLive)
	if heapLive == 0 {
		// TODO: this is a kludge. The metrics behind m.HeapLive aren't
		// updated until mark termination, so they'll be zero if we
		// haven't seen one yet. Use the profiler's own estimate of the
		// live heap here. Since m.HeapLive won't be included in
		// goMemory, but live heap allocations will likely contribute to
		// total RSS, add our estimation here to avoid double-counting
		// the live heap size in the samples we emit and in the non-Go
		// memory estimation.
		heapLive = liveWeightTotal
		goMemory += uint64(heapLive)
	}
	heapDead := max(int64(m.MemoryClassesHeapObjects)-heapLive, 0)
	if liveWeightTotal == 0 {
		emitSample(heapLive, livePath, nil, 0)
	}
	if deadWeightTotal == 0 {
		emitSample(heapDead, deadPath, nil, 0)
	}

	emitSample(int64(m.MemoryClassesHeapUnused), []rssVirtualFrame{rssFrameGoMemory, rssFrameHeap, rssFrameUnusedHeap}, nil, 0)
	emitSample(int64(m.MemoryClassesHeapFree), []rssVirtualFrame{rssFrameGoMemory, rssFrameHeap, rssFrameFreeHeap}, nil, 0)

	// Distribute each heap total in proportion to the sampled weights.
	// In words: The sampled values sum to W, and we want to scale them to
	// sum to W'. We could just multiply every value by (W'/W). But we want
	// to end up with an integer and can come up short of W' after rounding.
	// Instead, if we say S_N is the sum of the first N values's weights,
	// then we assign value N a new weight
	//     floor(S_N * W' / W) - floor(S_{N-1} * W' / W)
	// Note that if we drop the "floor"s in the expression we get exactly the
	// weight we want for value N if we allowed fractional values.
	// Since X - floor(X) is less than one, the absolute difference between
	// the weight we assign and the ideal weight is at most one for each value.
	// The last weight we assign is floor(W * W' / W) - floor(S_{N-1} * W' / W),
	// or W' - floor(S_{N-1} * W' / W). Working backwards we see that the total
	// weight assigned is W'.
	var liveCumWeight, deadCumWeight, liveAssigned, deadAssigned int64
	var realLocs []uint64
	for i, w := range weights {
		var liveValue, deadValue int64
		if w.live > 0 {
			liveCumWeight += w.live
			target := scaleHeapWeight(heapLive, liveCumWeight, liveWeightTotal)
			liveValue = target - liveAssigned
			liveAssigned = target
		}
		if w.dead > 0 {
			deadCumWeight += w.dead
			target := scaleHeapWeight(heapDead, deadCumWeight, deadWeightTotal)
			deadValue = target - deadAssigned
			deadAssigned = target
		}
		if liveValue == 0 && deadValue == 0 {
			continue
		}
		realLocs = b.appendLocsForStackWithoutRuntime(realLocs[:0], p[i].Stack)
		emitSample(liveValue, livePath, realLocs, p[i].ObjectSize)
		emitSample(deadValue, deadPath, realLocs, p[i].ObjectSize)
	}

	switch {
	case m.RSS == 0:
		b.pb.int64(tagProfile_Comment, b.stringIndex("RSS is unavailable on this system"))
	case m.RSS <= goMemory:
		b.pb.int64(tagProfile_Comment, b.stringIndex(fmt.Sprintf("RSS (%d bytes) is less than or equal to total Go memory (%d bytes)", m.RSS, goMemory)))
	case m.RSS > goMemory:
		emitSample(int64(m.RSS-goMemory), []rssVirtualFrame{rssFrameNonGoMemory}, nil, 0)
	}
	return b.build()
}

// scaleHeapWeight returns floor(total * weight / weightTotal) using a 128-bit
// product to avoid overflow. It requires total >= 0, 0 <= weight <= weightTotal,
// and weightTotal > 0. The quotient is at most total, so it fits in int64.
func scaleHeapWeight(total, weight, weightTotal int64) int64 {
	hi, lo := bits.Mul64(uint64(total), uint64(weight))
	q, _ := bits.Div64(hi, lo, uint64(weightTotal))
	return int64(q)
}

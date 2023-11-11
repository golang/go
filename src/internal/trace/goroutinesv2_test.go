// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trace

import (
	tracev2 "internal/trace/v2"
	"internal/trace/v2/testtrace"
	"testing"
)

func TestSummarizeGoroutinesTrace(t *testing.T) {
	summaries := summarizeTraceTest(t, "v2/testdata/tests/go122-gc-stress.test")
	var (
		hasSchedWaitTime    bool
		hasSyncBlockTime    bool
		hasGCMarkAssistTime bool
	)
	for _, summary := range summaries {
		basicSummaryChecks(t, summary)
		hasSchedWaitTime = hasSchedWaitTime || summary.SchedWaitTime > 0
		if dt, ok := summary.BlockTimeByReason["sync"]; ok && dt > 0 {
			hasSyncBlockTime = true
		}
		if dt, ok := summary.RangeTime["GC mark assist"]; ok && dt > 0 {
			hasGCMarkAssistTime = true
		}
	}
	if !hasSchedWaitTime {
		t.Error("missing sched wait time")
	}
	if !hasSyncBlockTime {
		t.Error("missing sync block time")
	}
	if !hasGCMarkAssistTime {
		t.Error("missing GC mark assist time")
	}
}

func TestSummarizeGoroutinesRegionsTrace(t *testing.T) {
	summaries := summarizeTraceTest(t, "v2/testdata/tests/go122-annotations.test")
	type region struct {
		startKind tracev2.EventKind
		endKind   tracev2.EventKind
	}
	wantRegions := map[string]region{
		// N.B. "pre-existing region" never even makes it into the trace.
		//
		// TODO(mknyszek): Add test case for end-without-a-start, which can happen at
		// a generation split only.
		"":                     {tracev2.EventStateTransition, tracev2.EventStateTransition}, // Task inheritance marker.
		"task0 region":         {tracev2.EventRegionBegin, tracev2.EventBad},
		"region0":              {tracev2.EventRegionBegin, tracev2.EventRegionEnd},
		"region1":              {tracev2.EventRegionBegin, tracev2.EventRegionEnd},
		"unended region":       {tracev2.EventRegionBegin, tracev2.EventStateTransition},
		"post-existing region": {tracev2.EventRegionBegin, tracev2.EventBad},
	}
	for _, summary := range summaries {
		basicSummaryChecks(t, summary)
		for _, region := range summary.Regions {
			want, ok := wantRegions[region.Name]
			if !ok {
				continue
			}
			checkRegionEvents(t, want.startKind, want.endKind, summary.ID, region)
			delete(wantRegions, region.Name)
		}
	}
	if len(wantRegions) != 0 {
		t.Errorf("failed to find regions: %#v", wantRegions)
	}
}

func basicSummaryChecks(t *testing.T, summary *GoroutineSummary) {
	if summary.ID == tracev2.NoGoroutine {
		t.Error("summary found for no goroutine")
		return
	}
	if (summary.StartTime != 0 && summary.CreationTime > summary.StartTime) ||
		(summary.StartTime != 0 && summary.EndTime != 0 && summary.StartTime > summary.EndTime) {
		t.Errorf("bad summary creation/start/end times for G %d: creation=%d start=%d end=%d", summary.ID, summary.CreationTime, summary.StartTime, summary.EndTime)
	}
	if (summary.PC != 0 && summary.Name == "") || (summary.PC == 0 && summary.Name != "") {
		t.Errorf("bad name and/or PC for G %d: pc=0x%x name=%q", summary.ID, summary.PC, summary.Name)
	}
	basicGoroutineExecStatsChecks(t, &summary.GoroutineExecStats)
	for _, region := range summary.Regions {
		basicGoroutineExecStatsChecks(t, &region.GoroutineExecStats)
	}
}

func summarizeTraceTest(t *testing.T, testPath string) map[tracev2.GoID]*GoroutineSummary {
	r, _, err := testtrace.ParseFile(testPath)
	if err != nil {
		t.Fatalf("malformed test %s: bad trace file: %v", testPath, err)
	}
	summaries, err := SummarizeGoroutines(r)
	if err != nil {
		t.Fatalf("failed to process trace %s: %v", testPath, err)
	}
	return summaries
}

func checkRegionEvents(t *testing.T, wantStart, wantEnd tracev2.EventKind, goid tracev2.GoID, region *UserRegionSummary) {
	switch wantStart {
	case tracev2.EventBad:
		if region.Start != nil {
			t.Errorf("expected nil region start event, got\n%s", region.Start.String())
		}
	case tracev2.EventStateTransition, tracev2.EventRegionBegin:
		if region.Start == nil {
			t.Error("expected non-nil region start event, got nil")
		}
		kind := region.Start.Kind()
		if kind != wantStart {
			t.Errorf("wanted region start event %s, got %s", wantStart, kind)
		}
		if kind == tracev2.EventRegionBegin {
			if region.Start.Region().Type != region.Name {
				t.Errorf("region name mismatch: event has %s, summary has %s", region.Start.Region().Type, region.Name)
			}
		} else {
			st := region.Start.StateTransition()
			if st.Resource.Kind != tracev2.ResourceGoroutine {
				t.Errorf("found region start event for the wrong resource: %s", st.Resource)
			}
			if st.Resource.Goroutine() != goid {
				t.Errorf("found region start event for the wrong resource: wanted goroutine %d, got %s", goid, st.Resource)
			}
			if old, _ := st.Goroutine(); old != tracev2.GoNotExist && old != tracev2.GoUndetermined {
				t.Errorf("expected transition from GoNotExist or GoUndetermined, got transition from %s instead", old)
			}
		}
	default:
		t.Errorf("unexpected want start event type: %s", wantStart)
	}

	switch wantEnd {
	case tracev2.EventBad:
		if region.End != nil {
			t.Errorf("expected nil region end event, got\n%s", region.End.String())
		}
	case tracev2.EventStateTransition, tracev2.EventRegionEnd:
		if region.End == nil {
			t.Error("expected non-nil region end event, got nil")
		}
		kind := region.End.Kind()
		if kind != wantEnd {
			t.Errorf("wanted region end event %s, got %s", wantEnd, kind)
		}
		if kind == tracev2.EventRegionEnd {
			if region.End.Region().Type != region.Name {
				t.Errorf("region name mismatch: event has %s, summary has %s", region.End.Region().Type, region.Name)
			}
		} else {
			st := region.End.StateTransition()
			if st.Resource.Kind != tracev2.ResourceGoroutine {
				t.Errorf("found region end event for the wrong resource: %s", st.Resource)
			}
			if st.Resource.Goroutine() != goid {
				t.Errorf("found region end event for the wrong resource: wanted goroutine %d, got %s", goid, st.Resource)
			}
			if _, new := st.Goroutine(); new != tracev2.GoNotExist {
				t.Errorf("expected transition to GoNotExist, got transition to %s instead", new)
			}
		}
	default:
		t.Errorf("unexpected want end event type: %s", wantEnd)
	}
}

func basicGoroutineExecStatsChecks(t *testing.T, stats *GoroutineExecStats) {
	if stats.ExecTime < 0 {
		t.Error("found negative ExecTime")
	}
	if stats.SchedWaitTime < 0 {
		t.Error("found negative SchedWaitTime")
	}
	if stats.SyscallTime < 0 {
		t.Error("found negative SyscallTime")
	}
	if stats.SyscallBlockTime < 0 {
		t.Error("found negative SyscallBlockTime")
	}
	if stats.TotalTime < 0 {
		t.Error("found negative TotalTime")
	}
	for reason, dt := range stats.BlockTimeByReason {
		if dt < 0 {
			t.Errorf("found negative BlockTimeByReason for %s", reason)
		}
	}
	for name, dt := range stats.RangeTime {
		if dt < 0 {
			t.Errorf("found negative RangeTime for range %s", name)
		}
	}
}

func TestRelatedGoroutinesV2Trace(t *testing.T) {
	testPath := "v2/testdata/tests/go122-gc-stress.test"
	r, _, err := testtrace.ParseFile(testPath)
	if err != nil {
		t.Fatalf("malformed test %s: bad trace file: %v", testPath, err)
	}
	targetg := tracev2.GoID(86)
	got, err := RelatedGoroutinesV2(r, targetg)
	if err != nil {
		t.Fatalf("failed to find related goroutines for %s: %v", testPath, err)
	}
	want := map[tracev2.GoID]struct{}{
		tracev2.GoID(86):  struct{}{}, // N.B. Result includes target.
		tracev2.GoID(85):  struct{}{},
		tracev2.GoID(111): struct{}{},
	}
	for goid := range got {
		if _, ok := want[goid]; ok {
			delete(want, goid)
		} else {
			t.Errorf("unexpected goroutine %d found in related goroutines for %d in test %s", goid, targetg, testPath)
		}
	}
	if len(want) != 0 {
		for goid := range want {
			t.Errorf("failed to find related goroutine %d for goroutine %d in test %s", goid, targetg, testPath)
		}
	}
}

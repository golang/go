// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trace_test

import (
	"internal/trace"
	"internal/trace/testtrace"
	"io"
	"testing"
)

func TestSummarizeGoroutinesTrace(t *testing.T) {
	summaries := summarizeTraceTest(t, "testdata/tests/go122-gc-stress.test").Goroutines
	var (
		hasSchedWaitTime    bool
		hasSyncBlockTime    bool
		hasGCMarkAssistTime bool
		hasUnknownTime      bool
	)

	assertContainsGoroutine(t, summaries, "runtime.gcBgMarkWorker")
	assertContainsGoroutine(t, summaries, "main.main.func1")

	for _, summary := range summaries {
		basicGoroutineSummaryChecks(t, summary)
		hasSchedWaitTime = hasSchedWaitTime || summary.SchedWaitTime > 0
		if dt, ok := summary.BlockTimeByReason["sync"]; ok && dt > 0 {
			hasSyncBlockTime = true
		}
		if dt, ok := summary.RangeTime["GC mark assist"]; ok && dt > 0 {
			hasGCMarkAssistTime = true
		}
		hasUnknownTime = hasUnknownTime || summary.UnknownTime() > 0
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
	if hasUnknownTime {
		t.Error("has time that is unaccounted for")
	}
}

func TestSummarizeGoroutinesRegionsTrace(t *testing.T) {
	summaries := summarizeTraceTest(t, "testdata/tests/go122-annotations.test").Goroutines
	type region struct {
		startKind trace.EventKind
		endKind   trace.EventKind
	}
	wantRegions := map[string]region{
		// N.B. "pre-existing region" never even makes it into the trace.
		//
		// TODO(mknyszek): Add test case for end-without-a-start, which can happen at
		// a generation split only.
		"":                     {trace.EventStateTransition, trace.EventStateTransition}, // Task inheritance marker.
		"task0 region":         {trace.EventRegionBegin, trace.EventBad},
		"region0":              {trace.EventRegionBegin, trace.EventRegionEnd},
		"region1":              {trace.EventRegionBegin, trace.EventRegionEnd},
		"unended region":       {trace.EventRegionBegin, trace.EventStateTransition},
		"post-existing region": {trace.EventRegionBegin, trace.EventBad},
	}
	for _, summary := range summaries {
		basicGoroutineSummaryChecks(t, summary)
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

func TestSummarizeTasksTrace(t *testing.T) {
	summaries := summarizeTraceTest(t, "testdata/tests/go122-annotations-stress.test").Tasks
	type task struct {
		name       string
		parent     *trace.TaskID
		children   []trace.TaskID
		logs       []trace.Log
		goroutines []trace.GoID
	}
	parent := func(id trace.TaskID) *trace.TaskID {
		p := new(trace.TaskID)
		*p = id
		return p
	}
	wantTasks := map[trace.TaskID]task{
		trace.BackgroundTask: {
			// The background task (0) is never any task's parent.
			logs: []trace.Log{
				{Task: trace.BackgroundTask, Category: "log", Message: "before do"},
				{Task: trace.BackgroundTask, Category: "log", Message: "before do"},
			},
			goroutines: []trace.GoID{1},
		},
		1: {
			// This started before tracing started and has no parents.
			// Task 2 is technically a child, but we lost that information.
			children: []trace.TaskID{3, 7, 16},
			logs: []trace.Log{
				{Task: 1, Category: "log", Message: "before do"},
				{Task: 1, Category: "log", Message: "before do"},
			},
			goroutines: []trace.GoID{1},
		},
		2: {
			// This started before tracing started and its parent is technically (1), but that information was lost.
			children: []trace.TaskID{8, 17},
			logs: []trace.Log{
				{Task: 2, Category: "log", Message: "before do"},
				{Task: 2, Category: "log", Message: "before do"},
			},
			goroutines: []trace.GoID{1},
		},
		3: {
			parent:   parent(1),
			children: []trace.TaskID{10, 19},
			logs: []trace.Log{
				{Task: 3, Category: "log", Message: "before do"},
				{Task: 3, Category: "log", Message: "before do"},
			},
			goroutines: []trace.GoID{1},
		},
		4: {
			// Explicitly, no parent.
			children: []trace.TaskID{12, 21},
			logs: []trace.Log{
				{Task: 4, Category: "log", Message: "before do"},
				{Task: 4, Category: "log", Message: "before do"},
			},
			goroutines: []trace.GoID{1},
		},
		12: {
			parent:   parent(4),
			children: []trace.TaskID{13},
			logs: []trace.Log{
				// TODO(mknyszek): This is computed asynchronously in the trace,
				// which makes regenerating this test very annoying, since it will
				// likely break this test. Resolve this by making the order not matter.
				{Task: 12, Category: "log2", Message: "do"},
				{Task: 12, Category: "log", Message: "fanout region4"},
				{Task: 12, Category: "log", Message: "fanout region0"},
				{Task: 12, Category: "log", Message: "fanout region1"},
				{Task: 12, Category: "log", Message: "fanout region2"},
				{Task: 12, Category: "log", Message: "before do"},
				{Task: 12, Category: "log", Message: "fanout region3"},
			},
			goroutines: []trace.GoID{1, 5, 6, 7, 8, 9},
		},
		13: {
			// Explicitly, no children.
			parent: parent(12),
			logs: []trace.Log{
				{Task: 13, Category: "log2", Message: "do"},
			},
			goroutines: []trace.GoID{7},
		},
	}
	for id, summary := range summaries {
		want, ok := wantTasks[id]
		if !ok {
			continue
		}
		if id != summary.ID {
			t.Errorf("ambiguous task %d (or %d?): field likely set incorrectly", id, summary.ID)
		}

		// Check parent.
		if want.parent != nil {
			if summary.Parent == nil {
				t.Errorf("expected parent %d for task %d without a parent", *want.parent, id)
			} else if summary.Parent.ID != *want.parent {
				t.Errorf("bad parent for task %d: want %d, got %d", id, *want.parent, summary.Parent.ID)
			}
		} else if summary.Parent != nil {
			t.Errorf("unexpected parent %d for task %d", summary.Parent.ID, id)
		}

		// Check children.
		gotChildren := make(map[trace.TaskID]struct{})
		for _, child := range summary.Children {
			gotChildren[child.ID] = struct{}{}
		}
		for _, wantChild := range want.children {
			if _, ok := gotChildren[wantChild]; ok {
				delete(gotChildren, wantChild)
			} else {
				t.Errorf("expected child task %d for task %d not found", wantChild, id)
			}
		}
		if len(gotChildren) != 0 {
			for child := range gotChildren {
				t.Errorf("unexpected child task %d for task %d", child, id)
			}
		}

		// Check logs.
		if len(want.logs) != len(summary.Logs) {
			t.Errorf("wanted %d logs for task %d, got %d logs instead", len(want.logs), id, len(summary.Logs))
		} else {
			for i := range want.logs {
				if want.logs[i] != summary.Logs[i].Log() {
					t.Errorf("log mismatch: want %#v, got %#v", want.logs[i], summary.Logs[i].Log())
				}
			}
		}

		// Check goroutines.
		if len(want.goroutines) != len(summary.Goroutines) {
			t.Errorf("wanted %d goroutines for task %d, got %d goroutines instead", len(want.goroutines), id, len(summary.Goroutines))
		} else {
			for _, goid := range want.goroutines {
				g, ok := summary.Goroutines[goid]
				if !ok {
					t.Errorf("want goroutine %d for task %d, not found", goid, id)
					continue
				}
				if g.ID != goid {
					t.Errorf("goroutine summary for %d does not match task %d listing of %d", g.ID, id, goid)
				}
			}
		}

		// Marked as seen.
		delete(wantTasks, id)
	}
	if len(wantTasks) != 0 {
		t.Errorf("failed to find tasks: %#v", wantTasks)
	}
}

func assertContainsGoroutine(t *testing.T, summaries map[trace.GoID]*trace.GoroutineSummary, name string) {
	for _, summary := range summaries {
		if summary.Name == name {
			return
		}
	}
	t.Errorf("missing goroutine %s", name)
}

func basicGoroutineSummaryChecks(t *testing.T, summary *trace.GoroutineSummary) {
	if summary.ID == trace.NoGoroutine {
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

func summarizeTraceTest(t *testing.T, testPath string) *trace.Summary {
	trc, _, _, err := testtrace.ParseFile(testPath)
	if err != nil {
		t.Fatalf("malformed test %s: bad trace file: %v", testPath, err)
	}
	// Create the analysis state.
	s := trace.NewSummarizer()

	// Create a reader.
	r, err := trace.NewReader(trc)
	if err != nil {
		t.Fatalf("failed to create trace reader for %s: %v", testPath, err)
	}
	// Process the trace.
	for {
		ev, err := r.ReadEvent()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("failed to process trace %s: %v", testPath, err)
		}
		s.Event(&ev)
	}
	return s.Finalize()
}

func checkRegionEvents(t *testing.T, wantStart, wantEnd trace.EventKind, goid trace.GoID, region *trace.UserRegionSummary) {
	switch wantStart {
	case trace.EventBad:
		if region.Start != nil {
			t.Errorf("expected nil region start event, got\n%s", region.Start.String())
		}
	case trace.EventStateTransition, trace.EventRegionBegin:
		if region.Start == nil {
			t.Error("expected non-nil region start event, got nil")
		}
		kind := region.Start.Kind()
		if kind != wantStart {
			t.Errorf("wanted region start event %s, got %s", wantStart, kind)
		}
		if kind == trace.EventRegionBegin {
			if region.Start.Region().Type != region.Name {
				t.Errorf("region name mismatch: event has %s, summary has %s", region.Start.Region().Type, region.Name)
			}
		} else {
			st := region.Start.StateTransition()
			if st.Resource.Kind != trace.ResourceGoroutine {
				t.Errorf("found region start event for the wrong resource: %s", st.Resource)
			}
			if st.Resource.Goroutine() != goid {
				t.Errorf("found region start event for the wrong resource: wanted goroutine %d, got %s", goid, st.Resource)
			}
			if old, _ := st.Goroutine(); old != trace.GoNotExist && old != trace.GoUndetermined {
				t.Errorf("expected transition from GoNotExist or GoUndetermined, got transition from %s instead", old)
			}
		}
	default:
		t.Errorf("unexpected want start event type: %s", wantStart)
	}

	switch wantEnd {
	case trace.EventBad:
		if region.End != nil {
			t.Errorf("expected nil region end event, got\n%s", region.End.String())
		}
	case trace.EventStateTransition, trace.EventRegionEnd:
		if region.End == nil {
			t.Error("expected non-nil region end event, got nil")
		}
		kind := region.End.Kind()
		if kind != wantEnd {
			t.Errorf("wanted region end event %s, got %s", wantEnd, kind)
		}
		if kind == trace.EventRegionEnd {
			if region.End.Region().Type != region.Name {
				t.Errorf("region name mismatch: event has %s, summary has %s", region.End.Region().Type, region.Name)
			}
		} else {
			st := region.End.StateTransition()
			if st.Resource.Kind != trace.ResourceGoroutine {
				t.Errorf("found region end event for the wrong resource: %s", st.Resource)
			}
			if st.Resource.Goroutine() != goid {
				t.Errorf("found region end event for the wrong resource: wanted goroutine %d, got %s", goid, st.Resource)
			}
			if _, new := st.Goroutine(); new != trace.GoNotExist {
				t.Errorf("expected transition to GoNotExist, got transition to %s instead", new)
			}
		}
	default:
		t.Errorf("unexpected want end event type: %s", wantEnd)
	}
}

func basicGoroutineExecStatsChecks(t *testing.T, stats *trace.GoroutineExecStats) {
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
	testPath := "testdata/tests/go122-gc-stress.test"
	trc, _, _, err := testtrace.ParseFile(testPath)
	if err != nil {
		t.Fatalf("malformed test %s: bad trace file: %v", testPath, err)
	}

	// Create a reader.
	r, err := trace.NewReader(trc)
	if err != nil {
		t.Fatalf("failed to create trace reader for %s: %v", testPath, err)
	}

	// Collect all the events.
	var events []trace.Event
	for {
		ev, err := r.ReadEvent()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("failed to process trace %s: %v", testPath, err)
		}
		events = append(events, ev)
	}

	// Test the function.
	targetg := trace.GoID(86)
	got := trace.RelatedGoroutinesV2(events, targetg)
	want := map[trace.GoID]struct{}{
		trace.GoID(86):  {}, // N.B. Result includes target.
		trace.GoID(71):  {},
		trace.GoID(25):  {},
		trace.GoID(122): {},
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

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !js

package main

import (
	"bytes"
	"context"
	"flag"
	"fmt"
	traceparser "internal/trace"
	"os"
	"reflect"
	"runtime/debug"
	"runtime/trace"
	"sort"
	"sync"
	"testing"
	"time"
)

var saveTraces = flag.Bool("savetraces", false, "save traces collected by tests")

func TestOverlappingDuration(t *testing.T) {
	cases := []struct {
		start0, end0, start1, end1 int64
		want                       time.Duration
	}{
		{
			1, 10, 11, 20, 0,
		},
		{
			1, 10, 5, 20, 5 * time.Nanosecond,
		},
		{
			1, 10, 2, 8, 6 * time.Nanosecond,
		},
	}

	for _, tc := range cases {
		s0, e0, s1, e1 := tc.start0, tc.end0, tc.start1, tc.end1
		if got := overlappingDuration(s0, e0, s1, e1); got != tc.want {
			t.Errorf("overlappingDuration(%d, %d, %d, %d)=%v; want %v", s0, e0, s1, e1, got, tc.want)
		}
		if got := overlappingDuration(s1, e1, s0, e0); got != tc.want {
			t.Errorf("overlappingDuration(%d, %d, %d, %d)=%v; want %v", s1, e1, s0, e0, got, tc.want)
		}
	}
}

// prog0 starts three goroutines.
//
//   goroutine 1: taskless region
//   goroutine 2: starts task0, do work in task0.region0, starts task1 which ends immediately.
//   goroutine 3: do work in task0.region1 and task0.region2, ends task0
func prog0() {
	ctx := context.Background()

	var wg sync.WaitGroup

	wg.Add(1)
	go func() { // goroutine 1
		defer wg.Done()
		trace.WithRegion(ctx, "taskless.region", func() {
			trace.Log(ctx, "key0", "val0")
		})
	}()

	wg.Add(1)
	go func() { // goroutine 2
		defer wg.Done()
		ctx, task := trace.NewTask(ctx, "task0")
		trace.WithRegion(ctx, "task0.region0", func() {
			wg.Add(1)
			go func() { // goroutine 3
				defer wg.Done()
				defer task.End()
				trace.WithRegion(ctx, "task0.region1", func() {
					trace.WithRegion(ctx, "task0.region2", func() {
						trace.Log(ctx, "key2", "val2")
					})
					trace.Log(ctx, "key1", "val1")
				})
			}()
		})
		ctx2, task2 := trace.NewTask(ctx, "task1")
		trace.Log(ctx2, "key3", "val3")
		task2.End()
	}()
	wg.Wait()
}

func TestAnalyzeAnnotations(t *testing.T) {
	// TODO: classify taskless regions

	// Run prog0 and capture the execution trace.
	if err := traceProgram(t, prog0, "TestAnalyzeAnnotations"); err != nil {
		t.Fatalf("failed to trace the program: %v", err)
	}

	res, err := analyzeAnnotations()
	if err != nil {
		t.Fatalf("failed to analyzeAnnotations: %v", err)
	}

	// For prog0, we expect
	//   - task with name = "task0", with three regions.
	//   - task with name = "task1", with no region.
	wantTasks := map[string]struct {
		complete   bool
		goroutines int
		regions    []string
	}{
		"task0": {
			complete:   true,
			goroutines: 2,
			regions:    []string{"task0.region0", "", "task0.region1", "task0.region2"},
		},
		"task1": {
			complete:   true,
			goroutines: 1,
		},
	}

	for _, task := range res.tasks {
		want, ok := wantTasks[task.name]
		if !ok {
			t.Errorf("unexpected task: %s", task)
			continue
		}
		if task.complete() != want.complete || len(task.goroutines) != want.goroutines || !reflect.DeepEqual(regionNames(task), want.regions) {
			t.Errorf("got task %v; want %+v", task, want)
		}

		delete(wantTasks, task.name)
	}
	if len(wantTasks) > 0 {
		t.Errorf("no more tasks; want %+v", wantTasks)
	}

	wantRegions := []string{
		"", // an auto-created region for the goroutine 3
		"taskless.region",
		"task0.region0",
		"task0.region1",
		"task0.region2",
	}
	var gotRegions []string
	for regionID := range res.regions {
		gotRegions = append(gotRegions, regionID.Type)
	}

	sort.Strings(wantRegions)
	sort.Strings(gotRegions)
	if !reflect.DeepEqual(gotRegions, wantRegions) {
		t.Errorf("got regions %q, want regions %q", gotRegions, wantRegions)
	}
}

// prog1 creates a task hierarchy consisting of three tasks.
func prog1() {
	ctx := context.Background()
	ctx1, task1 := trace.NewTask(ctx, "task1")
	defer task1.End()
	trace.WithRegion(ctx1, "task1.region", func() {
		ctx2, task2 := trace.NewTask(ctx1, "task2")
		defer task2.End()
		trace.WithRegion(ctx2, "task2.region", func() {
			ctx3, task3 := trace.NewTask(ctx2, "task3")
			defer task3.End()
			trace.WithRegion(ctx3, "task3.region", func() {
			})
		})
	})
}

func TestAnalyzeAnnotationTaskTree(t *testing.T) {
	// Run prog1 and capture the execution trace.
	if err := traceProgram(t, prog1, "TestAnalyzeAnnotationTaskTree"); err != nil {
		t.Fatalf("failed to trace the program: %v", err)
	}

	res, err := analyzeAnnotations()
	if err != nil {
		t.Fatalf("failed to analyzeAnnotations: %v", err)
	}
	tasks := res.tasks

	// For prog0, we expect
	//   - task with name = "", with taskless.region in regions.
	//   - task with name = "task0", with three regions.
	wantTasks := map[string]struct {
		parent   string
		children []string
		regions  []string
	}{
		"task1": {
			parent:   "",
			children: []string{"task2"},
			regions:  []string{"task1.region"},
		},
		"task2": {
			parent:   "task1",
			children: []string{"task3"},
			regions:  []string{"task2.region"},
		},
		"task3": {
			parent:   "task2",
			children: nil,
			regions:  []string{"task3.region"},
		},
	}

	for _, task := range tasks {
		want, ok := wantTasks[task.name]
		if !ok {
			t.Errorf("unexpected task: %s", task)
			continue
		}
		delete(wantTasks, task.name)

		if parentName(task) != want.parent ||
			!reflect.DeepEqual(childrenNames(task), want.children) ||
			!reflect.DeepEqual(regionNames(task), want.regions) {
			t.Errorf("got %v; want %+v", task, want)
		}
	}

	if len(wantTasks) > 0 {
		t.Errorf("no more tasks; want %+v", wantTasks)
	}
}

// prog2 starts two tasks; "taskWithGC" that overlaps with GC
// and "taskWithoutGC" that doesn't. In order to run this reliably,
// the caller needs to set up to prevent GC from running automatically.
// prog2 returns the upper-bound gc time that overlaps with the first task.
func prog2() (gcTime time.Duration) {
	ch := make(chan bool)
	ctx1, task := trace.NewTask(context.Background(), "taskWithGC")
	trace.WithRegion(ctx1, "taskWithGC.region1", func() {
		go func() {
			defer trace.StartRegion(ctx1, "taskWithGC.region2").End()
			<-ch
		}()
		s := time.Now()
		debug.FreeOSMemory() // task1 affected by gc
		gcTime = time.Since(s)
		close(ch)
	})
	task.End()

	ctx2, task2 := trace.NewTask(context.Background(), "taskWithoutGC")
	trace.WithRegion(ctx2, "taskWithoutGC.region1", func() {
		// do nothing.
	})
	task2.End()
	return gcTime
}

func TestAnalyzeAnnotationGC(t *testing.T) {
	err := traceProgram(t, func() {
		oldGC := debug.SetGCPercent(10000) // gc, and effectively disable GC
		defer debug.SetGCPercent(oldGC)
		prog2()
	}, "TestAnalyzeAnnotationGC")
	if err != nil {
		t.Fatalf("failed to trace the program: %v", err)
	}

	res, err := analyzeAnnotations()
	if err != nil {
		t.Fatalf("failed to analyzeAnnotations: %v", err)
	}

	// Check collected GC Start events are all sorted and non-overlapping.
	lastTS := int64(0)
	for i, ev := range res.gcEvents {
		if ev.Type != traceparser.EvGCStart {
			t.Errorf("unwanted event in gcEvents: %v", ev)
		}
		if i > 0 && lastTS > ev.Ts {
			t.Errorf("overlapping GC events:\n%d: %v\n%d: %v", i-1, res.gcEvents[i-1], i, res.gcEvents[i])
		}
		if ev.Link != nil {
			lastTS = ev.Link.Ts
		}
	}

	// Check whether only taskWithGC reports overlapping duration.
	for _, task := range res.tasks {
		got := task.overlappingGCDuration(res.gcEvents)
		switch task.name {
		case "taskWithoutGC":
			if got != 0 {
				t.Errorf("%s reported %v as overlapping GC time; want 0: %v", task.name, got, task)
			}
		case "taskWithGC":
			upperBound := task.duration()
			// TODO(hyangah): a tighter upper bound is gcTime, but
			// use of it will make the test flaky due to the issue
			// described in golang.org/issue/16755. Tighten the upper
			// bound when the issue with the timestamp computed
			// based on clockticks is resolved.
			if got <= 0 || got > upperBound {
				t.Errorf("%s reported %v as overlapping GC time; want (0, %v):\n%v", task.name, got, upperBound, task)
				buf := new(bytes.Buffer)
				fmt.Fprintln(buf, "GC Events")
				for _, ev := range res.gcEvents {
					fmt.Fprintf(buf, " %s -> %s\n", ev, ev.Link)
				}
				fmt.Fprintln(buf, "Events in Task")
				for i, ev := range task.events {
					fmt.Fprintf(buf, " %d: %s\n", i, ev)
				}

				t.Logf("\n%s", buf)
			}
		}
	}
}

// traceProgram runs the provided function while tracing is enabled,
// parses the captured trace, and sets the global trace loader to
// point to the parsed trace.
//
// If savetraces flag is set, the captured trace will be saved in the named file.
func traceProgram(t *testing.T, f func(), name string) error {
	t.Helper()
	buf := new(bytes.Buffer)
	if err := trace.Start(buf); err != nil {
		return err
	}
	f()
	trace.Stop()

	saveTrace(buf, name)
	res, err := traceparser.Parse(buf, name+".faketrace")
	if err == traceparser.ErrTimeOrder {
		t.Skipf("skipping due to golang.org/issue/16755: %v", err)
	} else if err != nil {
		return err
	}

	swapLoaderData(res, err)
	return nil
}

func regionNames(task *taskDesc) (ret []string) {
	for _, s := range task.regions {
		ret = append(ret, s.Name)
	}
	return ret
}

func parentName(task *taskDesc) string {
	if task.parent != nil {
		return task.parent.name
	}
	return ""
}

func childrenNames(task *taskDesc) (ret []string) {
	for _, s := range task.children {
		ret = append(ret, s.name)
	}
	return ret
}

func swapLoaderData(res traceparser.ParseResult, err error) {
	// swap loader's data.
	parseTrace() // fool loader.once.

	loader.res = res
	loader.err = err

	analyzeGoroutines(nil) // fool gsInit once.
	gs = traceparser.GoroutineStats(res.Events)

}

func saveTrace(buf *bytes.Buffer, name string) {
	if !*saveTraces {
		return
	}
	if err := os.WriteFile(name+".trace", buf.Bytes(), 0600); err != nil {
		panic(fmt.Errorf("failed to write trace file: %v", err))
	}
}

package main

import (
	"bytes"
	"context"
	"flag"
	"fmt"
	traceparser "internal/trace"
	"io/ioutil"
	"reflect"
	"runtime/debug"
	"runtime/trace"
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
//   goroutine 1: taskless span
//   goroutine 2: starts task0, do work in task0.span0, starts task1 which ends immediately.
//   goroutine 3: do work in task0.span1 and task0.span2, ends task0
func prog0() {
	ctx := context.Background()

	var wg sync.WaitGroup

	wg.Add(1)
	go func() { // goroutine 1
		defer wg.Done()
		trace.WithSpan(ctx, "taskless.span", func(ctx context.Context) {
			trace.Log(ctx, "key0", "val0")
		})
	}()

	wg.Add(1)
	go func() { // goroutine 2
		defer wg.Done()
		ctx, taskDone := trace.NewContext(ctx, "task0")
		trace.WithSpan(ctx, "task0.span0", func(ctx context.Context) {
			wg.Add(1)
			go func() { // goroutine 3
				defer wg.Done()
				defer taskDone()
				trace.WithSpan(ctx, "task0.span1", func(ctx context.Context) {
					trace.WithSpan(ctx, "task0.span2", func(ctx context.Context) {
						trace.Log(ctx, "key2", "val2")
					})
					trace.Log(ctx, "key1", "val1")
				})
			}()
		})
		ctx2, taskDone2 := trace.NewContext(ctx, "task1")
		trace.Log(ctx2, "key3", "val3")
		taskDone2()
	}()
	wg.Wait()
}

func TestAnalyzeAnnotations(t *testing.T) {
	// TODO: classify taskless spans

	// Run prog0 and capture the execution trace.
	if err := traceProgram(t, prog0, "TestAnalyzeAnnotations"); err != nil {
		t.Fatalf("failed to trace the program: %v", err)
	}

	res, err := analyzeAnnotations()
	if err != nil {
		t.Fatalf("failed to analyzeAnnotations: %v", err)
	}
	tasks := res.tasks

	// For prog0, we expect
	//   - task with name = "task0", with three spans.
	//   - task with name = "task1", with no span.
	wantTasks := map[string]struct {
		complete   bool
		goroutines int
		spans      []string
	}{
		"task0": {
			complete:   true,
			goroutines: 2,
			spans:      []string{"task0.span0", "task0.span1", "task0.span2"},
		},
		"task1": {
			complete:   true,
			goroutines: 1,
		},
	}

	for _, task := range tasks {
		want, ok := wantTasks[task.name]
		if !ok {
			t.Errorf("unexpected task: %s", task)
			continue
		}
		if task.complete() != want.complete || len(task.goroutines) != want.goroutines || !reflect.DeepEqual(spanNames(task), want.spans) {
			t.Errorf("got %v; want %+v", task, want)
		}

		delete(wantTasks, task.name)
	}
	if len(wantTasks) > 0 {
		t.Errorf("no more tasks; want %+v", wantTasks)
	}
}

// prog1 creates a task hierarchy consisting of three tasks.
func prog1() {
	ctx := context.Background()
	ctx1, done1 := trace.NewContext(ctx, "task1")
	defer done1()
	trace.WithSpan(ctx1, "task1.span", func(ctx context.Context) {
		ctx2, done2 := trace.NewContext(ctx, "task2")
		defer done2()
		trace.WithSpan(ctx2, "task2.span", func(ctx context.Context) {
			ctx3, done3 := trace.NewContext(ctx, "task3")
			defer done3()
			trace.WithSpan(ctx3, "task3.span", func(ctx context.Context) {
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
	//   - task with name = "", with taskless.span in spans.
	//   - task with name = "task0", with three spans.
	wantTasks := map[string]struct {
		parent   string
		children []string
		spans    []string
	}{
		"task1": {
			parent:   "",
			children: []string{"task2"},
			spans:    []string{"task1.span"},
		},
		"task2": {
			parent:   "task1",
			children: []string{"task3"},
			spans:    []string{"task2.span"},
		},
		"task3": {
			parent:   "task2",
			children: nil,
			spans:    []string{"task3.span"},
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
			!reflect.DeepEqual(spanNames(task), want.spans) {
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
	ctx1, done := trace.NewContext(context.Background(), "taskWithGC")
	trace.WithSpan(ctx1, "taskWithGC.span1", func(ctx context.Context) {
		go func() {
			defer trace.StartSpan(ctx, "taskWithGC.span2")()
			<-ch
		}()
		s := time.Now()
		debug.FreeOSMemory() // task1 affected by gc
		gcTime = time.Since(s)
		close(ch)
	})
	done()

	ctx2, done2 := trace.NewContext(context.Background(), "taskWithoutGC")
	trace.WithSpan(ctx2, "taskWithoutGC.span1", func(ctx context.Context) {
		// do nothing.
	})
	done2()
	return gcTime
}

func TestAnalyzeAnnotationGC(t *testing.T) {
	var gcTime time.Duration
	err := traceProgram(t, func() {
		oldGC := debug.SetGCPercent(10000) // gc, and effectively disable GC
		defer debug.SetGCPercent(oldGC)

		gcTime = prog2()
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

func spanNames(task *taskDesc) (ret []string) {
	for _, s := range task.spans {
		ret = append(ret, s.name)
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
}

func saveTrace(buf *bytes.Buffer, name string) {
	if !*saveTraces {
		return
	}
	if err := ioutil.WriteFile(name+".trace", buf.Bytes(), 0600); err != nil {
		panic(fmt.Errorf("failed to write trace file: %v", err))
	}
}

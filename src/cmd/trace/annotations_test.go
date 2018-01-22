package main

import (
	"bytes"
	"context"
	"flag"
	"fmt"
	traceparser "internal/trace"
	"io/ioutil"
	"reflect"
	"runtime/trace"
	"sync"
	"testing"
)

var saveTraces = flag.Bool("savetraces", false, "save traces collected by tests")

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
	traceProgram(prog0, "TestAnalyzeAnnotations")

	tasks, err := analyzeAnnotations()
	if err != nil {
		t.Fatalf("failed to analyzeAnnotations: %v", err)
	}
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

// traceProgram runs the provided function while tracing is enabled,
// parses the captured trace, and sets the global trace loader to
// point to the parsed trace.
//
// If savetraces flag is set, the captured trace will be saved in the named file.
func traceProgram(f func(), name string) error {
	buf := new(bytes.Buffer)
	if err := trace.Start(buf); err != nil {
		return err
	}
	f()
	trace.Stop()

	saveTrace(buf, name)
	res, err := traceparser.Parse(buf, "")
	if err != nil {
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

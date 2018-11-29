package trace_test

import (
	"bytes"
	"context"
	"fmt"
	"internal/trace"
	"reflect"
	. "runtime/trace"
	"strings"
	"sync"
	"testing"
)

func BenchmarkStartRegion(b *testing.B) {
	b.ReportAllocs()
	ctx, task := NewTask(context.Background(), "benchmark")
	defer task.End()

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			StartRegion(ctx, "region").End()
		}
	})
}

func BenchmarkNewTask(b *testing.B) {
	b.ReportAllocs()
	pctx, task := NewTask(context.Background(), "benchmark")
	defer task.End()

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			_, task := NewTask(pctx, "task")
			task.End()
		}
	})
}

func TestUserTaskRegion(t *testing.T) {
	if IsEnabled() {
		t.Skip("skipping because -test.trace is set")
	}
	bgctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	preExistingRegion := StartRegion(bgctx, "pre-existing region")

	buf := new(bytes.Buffer)
	if err := Start(buf); err != nil {
		t.Fatalf("failed to start tracing: %v", err)
	}

	// Beginning of traced execution
	var wg sync.WaitGroup
	ctx, task := NewTask(bgctx, "task0") // EvUserTaskCreate("task0")
	wg.Add(1)
	go func() {
		defer wg.Done()
		defer task.End() // EvUserTaskEnd("task0")

		WithRegion(ctx, "region0", func() {
			// EvUserRegionCreate("region0", start)
			WithRegion(ctx, "region1", func() {
				Log(ctx, "key0", "0123456789abcdef") // EvUserLog("task0", "key0", "0....f")
			})
			// EvUserRegion("region0", end)
		})
	}()

	wg.Wait()

	preExistingRegion.End()
	postExistingRegion := StartRegion(bgctx, "post-existing region")

	// End of traced execution
	Stop()

	postExistingRegion.End()

	saveTrace(t, buf, "TestUserTaskRegion")
	res, err := trace.Parse(buf, "")
	if err == trace.ErrTimeOrder {
		// golang.org/issues/16755
		t.Skipf("skipping trace: %v", err)
	}
	if err != nil {
		t.Fatalf("Parse failed: %v", err)
	}

	// Check whether we see all user annotation related records in order
	type testData struct {
		typ     byte
		strs    []string
		args    []uint64
		setLink bool
	}

	var got []testData
	tasks := map[uint64]string{}
	for _, e := range res.Events {
		t.Logf("%s", e)
		switch e.Type {
		case trace.EvUserTaskCreate:
			taskName := e.SArgs[0]
			got = append(got, testData{trace.EvUserTaskCreate, []string{taskName}, nil, e.Link != nil})
			if e.Link != nil && e.Link.Type != trace.EvUserTaskEnd {
				t.Errorf("Unexpected linked event %q->%q", e, e.Link)
			}
			tasks[e.Args[0]] = taskName
		case trace.EvUserLog:
			key, val := e.SArgs[0], e.SArgs[1]
			taskName := tasks[e.Args[0]]
			got = append(got, testData{trace.EvUserLog, []string{taskName, key, val}, nil, e.Link != nil})
		case trace.EvUserTaskEnd:
			taskName := tasks[e.Args[0]]
			got = append(got, testData{trace.EvUserTaskEnd, []string{taskName}, nil, e.Link != nil})
			if e.Link != nil && e.Link.Type != trace.EvUserTaskCreate {
				t.Errorf("Unexpected linked event %q->%q", e, e.Link)
			}
		case trace.EvUserRegion:
			taskName := tasks[e.Args[0]]
			regionName := e.SArgs[0]
			got = append(got, testData{trace.EvUserRegion, []string{taskName, regionName}, []uint64{e.Args[1]}, e.Link != nil})
			if e.Link != nil && (e.Link.Type != trace.EvUserRegion || e.Link.SArgs[0] != regionName) {
				t.Errorf("Unexpected linked event %q->%q", e, e.Link)
			}
		}
	}
	want := []testData{
		{trace.EvUserTaskCreate, []string{"task0"}, nil, true},
		{trace.EvUserRegion, []string{"task0", "region0"}, []uint64{0}, true},
		{trace.EvUserRegion, []string{"task0", "region1"}, []uint64{0}, true},
		{trace.EvUserLog, []string{"task0", "key0", "0123456789abcdef"}, nil, false},
		{trace.EvUserRegion, []string{"task0", "region1"}, []uint64{1}, false},
		{trace.EvUserRegion, []string{"task0", "region0"}, []uint64{1}, false},
		{trace.EvUserTaskEnd, []string{"task0"}, nil, false},
		//  Currently, pre-existing region is not recorded to avoid allocations.
		//  {trace.EvUserRegion, []string{"", "pre-existing region"}, []uint64{1}, false},
		{trace.EvUserRegion, []string{"", "post-existing region"}, []uint64{0}, false},
	}
	if !reflect.DeepEqual(got, want) {
		pretty := func(data []testData) string {
			var s strings.Builder
			for _, d := range data {
				s.WriteString(fmt.Sprintf("\t%+v\n", d))
			}
			return s.String()
		}
		t.Errorf("Got user region related events\n%+v\nwant:\n%+v", pretty(got), pretty(want))
	}
}

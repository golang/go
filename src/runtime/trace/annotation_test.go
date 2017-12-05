package trace_test

import (
	"bytes"
	"context"
	"internal/trace"
	"reflect"
	. "runtime/trace"
	"sync"
	"testing"
)

func TestUserTaskSpan(t *testing.T) {
	bgctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// TODO(hyangah): test pre-existing spans don't cause troubles

	buf := new(bytes.Buffer)
	if err := Start(buf); err != nil {
		t.Fatalf("failed to start tracing: %v", err)
	}

	// Beginning of traced execution
	var wg sync.WaitGroup
	ctx, end := NewContext(bgctx, "task0") // EvUserTaskCreate("task0")
	wg.Add(1)
	go func() {
		defer wg.Done()
		defer end() // EvUserTaskEnd("span0")

		WithSpan(ctx, "span0", func(ctx context.Context) {
			// EvUserSpanCreate("span0", start)
			Log(ctx, "key0", "0123456789abcdef") // EvUserLog("task0", "key0", "0....f")
			// EvUserSpan("span0", end)
		})
	}()
	wg.Wait()
	// End of traced execution
	Stop()
	saveTrace(t, buf, "TestUserTaskSpan")
	res, err := trace.Parse(buf, "")
	if err != nil {
		t.Fatalf("Parse failed: %v", err)
	}

	// Check whether we see all user annotation related records in order
	type testData struct {
		typ  byte
		strs []string
		args []uint64
	}

	var got []testData
	tasks := map[uint64]string{}
	for _, e := range res.Events {
		t.Logf("%s", e)
		switch e.Type {
		case trace.EvUserTaskCreate:
			taskName := e.SArgs[0]
			got = append(got, testData{trace.EvUserTaskCreate, []string{taskName}, nil})
			tasks[e.Args[0]] = taskName
		case trace.EvUserLog:
			key, val := e.SArgs[0], e.SArgs[1]
			taskName := tasks[e.Args[0]]
			got = append(got, testData{trace.EvUserLog, []string{taskName, key, val}, nil})
		case trace.EvUserTaskEnd:
			taskName := tasks[e.Args[0]]
			got = append(got, testData{trace.EvUserTaskEnd, []string{taskName}, nil})
		case trace.EvUserSpan:
			taskName := tasks[e.Args[0]]
			spanName := e.SArgs[0]
			got = append(got, testData{trace.EvUserSpan, []string{taskName, spanName}, []uint64{e.Args[1]}})
		}
	}
	want := []testData{
		{trace.EvUserTaskCreate, []string{"task0"}, nil},
		{trace.EvUserSpan, []string{"task0", "span0"}, []uint64{0}},
		{trace.EvUserLog, []string{"task0", "key0", "0123456789abcdef"}, nil},
		{trace.EvUserSpan, []string{"task0", "span0"}, []uint64{1}},
		{trace.EvUserTaskEnd, []string{"task0"}, nil},
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Got user span related events %+v\nwant: %+v", got, want)
	}
}

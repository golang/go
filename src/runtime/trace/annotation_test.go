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

func TestUserTaskSpan(t *testing.T) {
	bgctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	preExistingSpanEnd := StartSpan(bgctx, "pre-existing span")

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
		defer end() // EvUserTaskEnd("task0")

		WithSpan(ctx, "span0", func(ctx context.Context) {
			// EvUserSpanCreate("span0", start)
			WithSpan(ctx, "span1", func(ctx context.Context) {
				Log(ctx, "key0", "0123456789abcdef") // EvUserLog("task0", "key0", "0....f")
			})
			// EvUserSpan("span0", end)
		})
	}()

	wg.Wait()

	preExistingSpanEnd()
	postExistingSpanEnd := StartSpan(bgctx, "post-existing span")

	// End of traced execution
	Stop()

	postExistingSpanEnd()

	saveTrace(t, buf, "TestUserTaskSpan")
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
		case trace.EvUserSpan:
			taskName := tasks[e.Args[0]]
			spanName := e.SArgs[0]
			got = append(got, testData{trace.EvUserSpan, []string{taskName, spanName}, []uint64{e.Args[1]}, e.Link != nil})
			if e.Link != nil && (e.Link.Type != trace.EvUserSpan || e.Link.SArgs[0] != spanName) {
				t.Errorf("Unexpected linked event %q->%q", e, e.Link)
			}
		}
	}
	want := []testData{
		{trace.EvUserTaskCreate, []string{"task0"}, nil, true},
		{trace.EvUserSpan, []string{"task0", "span0"}, []uint64{0}, true},
		{trace.EvUserSpan, []string{"task0", "span1"}, []uint64{0}, true},
		{trace.EvUserLog, []string{"task0", "key0", "0123456789abcdef"}, nil, false},
		{trace.EvUserSpan, []string{"task0", "span1"}, []uint64{1}, false},
		{trace.EvUserSpan, []string{"task0", "span0"}, []uint64{1}, false},
		{trace.EvUserTaskEnd, []string{"task0"}, nil, false},
		{trace.EvUserSpan, []string{"", "pre-existing span"}, []uint64{1}, false},
		{trace.EvUserSpan, []string{"", "post-existing span"}, []uint64{0}, false},
	}
	if !reflect.DeepEqual(got, want) {
		pretty := func(data []testData) string {
			var s strings.Builder
			for _, d := range data {
				s.WriteString(fmt.Sprintf("\t%+v\n", d))
			}
			return s.String()
		}
		t.Errorf("Got user span related events\n%+v\nwant:\n%+v", pretty(got), pretty(want))
	}
}

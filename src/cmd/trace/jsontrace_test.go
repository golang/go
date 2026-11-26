// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"html"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"runtime/trace"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	itrace "internal/trace"
	"internal/trace/raw"
	"internal/trace/traceviewer"
	"internal/trace/traceviewer/format"
)

func TestJSONTraceRegions(t *testing.T) {
	var buf bytes.Buffer
	if err := trace.Start(&buf); err != nil {
		t.Fatal(err)
	}
	ctx := t.Context()
	var wg sync.WaitGroup
	trace.Log(ctx, "", "before")
	read := trace.StartRegion(ctx, "stage/read")
	wg.Go(func() { trace.Log(ctx, "", "unrelated goroutine") })
	wg.Wait()
	read.End()
	trace.Log(ctx, "", "between")
	wg.Go(func() {
		// A region can be selected even if its task never ends.
		ctx, _ := trace.NewTask(ctx, "request")
		write := trace.StartRegion(ctx, "stage/write")
		defer write.End()
		trace.WithRegion(ctx, "other", func() { trace.Log(ctx, "", "writing") })
	})
	wg.Wait()
	trace.Log(ctx, "", "after")
	trace.Stop()

	parsed, err := parseTrace(&buf, int64(buf.Len()))
	if err != nil {
		t.Fatal(err)
	}
	get := func(h http.Handler, target string) *httptest.ResponseRecorder {
		w := httptest.NewRecorder()
		h.ServeHTTP(w, httptest.NewRequest("GET", target, nil))
		return w
	}
	pattern := "^stage/(read|write)$"
	page := get(UserRegionsHandlerFunc(parsed), "/userregions?region="+url.QueryEscape(pattern))
	href := regexp.MustCompile(`href="(/trace\?[^"]+)"`).FindStringSubmatch(page.Body.String())
	if page.Code != http.StatusOK || href == nil {
		t.Fatalf("region search returned status %d, timeline link %v", page.Code, href)
	}
	link, err := url.Parse(html.UnescapeString(href[1]))
	if err != nil {
		t.Fatal(err)
	}
	// The link should zoom to the beginning of the first region and the end
	// of the last, in milliseconds relative to the start of the trace.
	when := func(kind itrace.EventKind, name string) float64 {
		for _, ev := range parsed.events {
			if ev.Kind() == kind && ev.Region().Type == name {
				return asMillisecond(ev.Time().Sub(parsed.startTime()))
			}
		}
		t.Fatalf("trace missing %s for %q", kind, name)
		return 0
	}
	wantSpan := fmt.Sprintf("%g:%g", when(itrace.EventRegionBegin, "stage/read"), when(itrace.EventRegionEnd, "stage/write"))
	if link.Query().Get("region") != pattern || link.Fragment != wantSpan {
		t.Errorf("timeline link = %s, want pattern %q and span %s", link, pattern, wantSpan)
	}
	h := JSONTraceHandler(parsed)
	w := get(h, "/jsontrace?"+link.RawQuery)
	if w.Code != http.StatusOK {
		t.Fatalf("query regions: %s", w.Result().Status)
	}
	var data format.Data
	if err := json.Unmarshal(w.Body.Bytes(), &data); err != nil {
		t.Fatal(err)
	}
	var regions, logs []string
	for _, ev := range data.Events {
		switch ev.Category {
		case "Region":
			regions = append(regions, ev.Name+"/"+ev.Phase)
		case "user event":
			logs = append(logs, ev.Name)
		}
	}
	slices.Sort(regions)
	if want := []string{"stage/read/b", "stage/read/e", "stage/write/b", "stage/write/e"}; !slices.Equal(regions, want) {
		t.Errorf("region events = %v, want %v", regions, want)
	}
	// Include events between matches on selected goroutines. Exclude events
	// outside that time window or on unrelated goroutines within it.
	if want := []string{"between", "writing"}; !slices.Equal(logs, want) {
		t.Errorf("logs = %v, want %v", logs, want)
	}

	for pattern, want := range map[string]int{"": http.StatusBadRequest, "[": http.StatusBadRequest, "^absent$": http.StatusNotFound} {
		if w := get(h, "/jsontrace?region="+url.QueryEscape(pattern)); w.Code != want || w.Body.Len() == 0 {
			t.Errorf("region %q: status %d, want %d and an error message", pattern, w.Code, want)
		}
	}
}

func TestJSONTraceHandler(t *testing.T) {
	testPaths, err := filepath.Glob("./testdata/*.test")
	if err != nil {
		t.Fatalf("discovering tests: %v", err)
	}
	for _, testPath := range testPaths {
		t.Run(filepath.Base(testPath), func(t *testing.T) {
			parsed := getTestTrace(t, testPath)
			data := recordJSONTraceHandlerResponse(t, parsed)
			// TODO(mknyszek): Check that there's one at most goroutine per proc at any given time.
			checkExecutionTimes(t, data)
			checkPlausibleHeapMetrics(t, data)
			// TODO(mknyszek): Check for plausible thread and goroutine metrics.
			checkMetaNamesEmitted(t, data, "process_name", []string{"STATS", "PROCS"})
			checkMetaNamesEmitted(t, data, "thread_name", []string{"GC", "Network", "Timers", "Syscalls", "Proc 0"})
			checkProcStartStop(t, data)
			checkSyscalls(t, data)
			checkNetworkUnblock(t, data)
			// TODO(mknyszek): Check for flow events.
		})
	}
}

func checkSyscalls(t *testing.T, data format.Data) {
	data = filterViewerTrace(data,
		filterEventName("syscall"),
		filterStackRootFunc("main.blockingSyscall"))
	if len(data.Events) <= 1 {
		t.Errorf("got %d events, want > 1", len(data.Events))
	}
	data = filterViewerTrace(data, filterBlocked("yes"))
	if len(data.Events) != 1 {
		t.Errorf("got %d events, want 1", len(data.Events))
	}
}

type eventFilterFn func(*format.Event, *format.Data) bool

func filterEventName(name string) eventFilterFn {
	return func(e *format.Event, _ *format.Data) bool {
		return e.Name == name
	}
}

// filterGoRoutineName returns an event filter that returns true if the event's
// goroutine name is equal to name.
func filterGoRoutineName(name string) eventFilterFn {
	return func(e *format.Event, _ *format.Data) bool {
		return parseGoroutineName(e) == name
	}
}

// parseGoroutineName returns the goroutine name from the event's name field.
// E.g. if e.Name is "G42 main.cpu10", this returns "main.cpu10".
func parseGoroutineName(e *format.Event) string {
	parts := strings.SplitN(e.Name, " ", 2)
	if len(parts) != 2 || !strings.HasPrefix(parts[0], "G") {
		return ""
	}
	return parts[1]
}

// filterBlocked returns an event filter that returns true if the event's
// "blocked" argument is equal to blocked.
func filterBlocked(blocked string) eventFilterFn {
	return func(e *format.Event, _ *format.Data) bool {
		m, ok := e.Arg.(map[string]any)
		if !ok {
			return false
		}
		return m["blocked"] == blocked
	}
}

// filterStackRootFunc returns an event filter that returns true if the function
// at the root of the stack trace is named name.
func filterStackRootFunc(name string) eventFilterFn {
	return func(e *format.Event, data *format.Data) bool {
		frames := stackFrames(data, e.Stack)
		rootFrame := frames[len(frames)-1]
		return strings.HasPrefix(rootFrame, name+":")
	}
}

// filterViewerTrace returns a copy of data with only the events that pass all
// of the given filters.
func filterViewerTrace(data format.Data, fns ...eventFilterFn) (filtered format.Data) {
	filtered = data
	filtered.Events = nil
	for _, e := range data.Events {
		keep := true
		for _, fn := range fns {
			keep = keep && fn(e, &filtered)
		}
		if keep {
			filtered.Events = append(filtered.Events, e)
		}
	}
	return
}

func stackFrames(data *format.Data, stackID int) (frames []string) {
	for {
		frame, ok := data.Frames[strconv.Itoa(stackID)]
		if !ok {
			return
		}
		frames = append(frames, frame.Name)
		stackID = frame.Parent
	}
}

func checkProcStartStop(t *testing.T, data format.Data) {
	procStarted := map[uint64]bool{}
	for _, e := range data.Events {
		if e.Name == "proc start" {
			if procStarted[e.TID] == true {
				t.Errorf("proc started twice: %d", e.TID)
			}
			procStarted[e.TID] = true
		}
		if e.Name == "proc stop" {
			if procStarted[e.TID] == false {
				t.Errorf("proc stopped twice: %d", e.TID)
			}
			procStarted[e.TID] = false
		}
	}
	if got, want := len(procStarted), 8; got != want {
		t.Errorf("wrong number of procs started/stopped got=%d want=%d", got, want)
	}
}

func checkNetworkUnblock(t *testing.T, data format.Data) {
	count := 0
	var netBlockEv *format.Event
	for _, e := range data.Events {
		if e.TID == traceviewer.NetpollP && e.Name == "unblock (network)" && e.Phase == "I" && e.Scope == "t" {
			count++
			netBlockEv = e
		}
	}
	if netBlockEv == nil {
		t.Error("failed to find a network unblock")
	}
	if count == 0 {
		t.Errorf("found zero network block events, want at least one")
	}
	// TODO(mknyszek): Check for the flow of this event to some slice event of a goroutine running.
}

func checkExecutionTimes(t *testing.T, data format.Data) {
	cpu10 := sumExecutionTime(filterViewerTrace(data, filterGoRoutineName("main.cpu10")))
	cpu20 := sumExecutionTime(filterViewerTrace(data, filterGoRoutineName("main.cpu20")))
	if cpu10 <= 0 || cpu20 <= 0 || cpu10 >= cpu20 {
		t.Errorf("bad execution times: cpu10=%v, cpu20=%v", cpu10, cpu20)
	}
}

func checkMetaNamesEmitted(t *testing.T, data format.Data, category string, want []string) {
	t.Helper()
	names := metaEventNameArgs(category, data)
	for _, wantName := range want {
		if !slices.Contains(names, wantName) {
			t.Errorf("%s: names=%v, want %q", category, names, wantName)
		}
	}
}

func metaEventNameArgs(category string, data format.Data) (names []string) {
	for _, e := range data.Events {
		if e.Name == category && e.Phase == "M" {
			names = append(names, e.Arg.(map[string]any)["name"].(string))
		}
	}
	return
}

func checkPlausibleHeapMetrics(t *testing.T, data format.Data) {
	hms := heapMetrics(data)
	var nonZeroAllocated, nonZeroNextGC bool
	for _, hm := range hms {
		if hm.Allocated > 0 {
			nonZeroAllocated = true
		}
		if hm.NextGC > 0 {
			nonZeroNextGC = true
		}
	}

	if !nonZeroAllocated {
		t.Errorf("nonZeroAllocated=%v, want true", nonZeroAllocated)
	}
	if !nonZeroNextGC {
		t.Errorf("nonZeroNextGC=%v, want true", nonZeroNextGC)
	}
}

func heapMetrics(data format.Data) (metrics []format.HeapCountersArg) {
	for _, e := range data.Events {
		if e.Phase == "C" && e.Name == "Heap" {
			j, _ := json.Marshal(e.Arg)
			var metric format.HeapCountersArg
			json.Unmarshal(j, &metric)
			metrics = append(metrics, metric)
		}
	}
	return
}

func recordJSONTraceHandlerResponse(t *testing.T, parsed *parsedTrace) format.Data {
	h := JSONTraceHandler(parsed)
	recorder := httptest.NewRecorder()
	r := httptest.NewRequest("GET", "/jsontrace", nil)
	h.ServeHTTP(recorder, r)

	var data format.Data
	if err := json.Unmarshal(recorder.Body.Bytes(), &data); err != nil {
		t.Fatal(err)
	}
	return data
}

func sumExecutionTime(data format.Data) (sum time.Duration) {
	for _, e := range data.Events {
		sum += time.Duration(e.Dur) * time.Microsecond
	}
	return
}

func getTestTrace(t *testing.T, testPath string) *parsedTrace {
	t.Helper()

	// First read in the text trace and write it out as bytes.
	f, err := os.Open(testPath)
	if err != nil {
		t.Fatalf("failed to open test %s: %v", testPath, err)
	}
	r, err := raw.NewTextReader(f)
	if err != nil {
		t.Fatalf("failed to read test %s: %v", testPath, err)
	}
	var trace bytes.Buffer
	w, err := raw.NewWriter(&trace, r.Version())
	if err != nil {
		t.Fatalf("failed to write out test %s: %v", testPath, err)
	}
	for {
		ev, err := r.ReadEvent()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("failed to read test %s: %v", testPath, err)
		}
		if err := w.WriteEvent(ev); err != nil {
			t.Fatalf("failed to write out test %s: %v", testPath, err)
		}
	}

	// Parse the test trace.
	parsed, err := parseTrace(&trace, int64(trace.Len()))
	if err != nil {
		t.Fatalf("failed to parse trace: %v", err)
	}
	return parsed
}

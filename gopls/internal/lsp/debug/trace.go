// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package debug

import (
	"bytes"
	"context"
	"fmt"
	"html/template"
	"net/http"
	"runtime/trace"
	"sort"
	"strings"
	"sync"
	"time"

	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/event/core"
	"golang.org/x/tools/internal/event/export"
	"golang.org/x/tools/internal/event/label"
)

// TraceTmpl extends BaseTemplate and renders a TraceResults, e.g. from getData().
var TraceTmpl = template.Must(template.Must(BaseTemplate.Clone()).Parse(`
{{define "title"}}Trace Information{{end}}
{{define "body"}}
	{{range .Traces}}<a href="/trace/{{.Name}}">{{.Name}}</a> last: {{.Last.Duration}}, longest: {{.Longest.Duration}}<br>{{end}}
	{{if .Selected}}
		<H2>{{.Selected.Name}}</H2>
		{{if .Selected.Last}}<H3>Last</H3><ul class='spans'>{{template "completeSpan" .Selected.Last}}</ul>{{end}}
		{{if .Selected.Longest}}<H3>Longest</H3><ul class='spans'>{{template "completeSpan" .Selected.Longest}}</ul>{{end}}
	{{end}}

        <H2>Recent spans (oldest first)</H2>
        <p>
        A finite number of recent span start/end times are shown below.
        The nesting represents the children of a parent span (and the log events within a span).
        A span may appear twice: chronologically at toplevel, and nested within its parent.
        </p>
	<ul class='spans'>{{range .Recent}}{{template "spanStartEnd" .}}{{end}}</ul>
{{end}}
{{define "spanStartEnd"}}
	{{if .Start}}
		<li>{{.Span.Header .Start}}</li>
	{{else}}
		{{template "completeSpan" .Span}}
	{{end}}
{{end}}
{{define "completeSpan"}}
	<li>{{.Header false}}</li>
	{{if .Events}}<ul>{{range .Events}}<li>{{.Header}}</li>{{end}}</ul>{{end}}
	{{if .ChildStartEnd}}<ul>{{range .ChildStartEnd}}{{template "spanStartEnd" .}}{{end}}</ul>{{end}}
{{end}}
`))

type traces struct {
	mu              sync.Mutex
	sets            map[string]*traceSet
	unfinished      map[export.SpanContext]*traceSpan
	recent          []spanStartEnd
	recentEvictions int
}

// A spanStartEnd records the start or end of a span.
// If Start, the span may be unfinished, so some fields (e.g. Finish)
// may be unset and others (e.g. Events) may be being actively populated.
type spanStartEnd struct {
	Start bool
	Span  *traceSpan
}

func (ev spanStartEnd) Time() time.Time {
	if ev.Start {
		return ev.Span.Start
	} else {
		return ev.Span.Finish
	}
}

// A TraceResults is the subject for the /trace HTML template.
type TraceResults struct { // exported for testing
	Traces   []*traceSet
	Selected *traceSet
	Recent   []spanStartEnd
}

// A traceSet holds two representative spans of a given span name.
type traceSet struct {
	Name    string
	Last    *traceSpan
	Longest *traceSpan
}

// A traceSpan holds information about a single span.
type traceSpan struct {
	TraceID       export.TraceID
	SpanID        export.SpanID
	ParentID      export.SpanID
	Name          string
	Start         time.Time
	Finish        time.Time     // set at end
	Duration      time.Duration // set at end
	Tags          string
	Events        []traceEvent   // set at end
	ChildStartEnd []spanStartEnd // populated while active

	parent *traceSpan
}

const timeFormat = "15:04:05.000"

// Header renders the time, name, tags, and (if !start),
// duration of a span start or end event.
func (span *traceSpan) Header(start bool) string {
	if start {
		return fmt.Sprintf("%s start %s %s",
			span.Start.Format(timeFormat), span.Name, span.Tags)
	} else {
		return fmt.Sprintf("%s end %s (+%s) %s",
			span.Finish.Format(timeFormat), span.Name, span.Duration, span.Tags)
	}
}

type traceEvent struct {
	Time   time.Time
	Offset time.Duration // relative to start of span
	Tags   string
}

func (ev traceEvent) Header() string {
	return fmt.Sprintf("%s event (+%s) %s", ev.Time.Format(timeFormat), ev.Offset, ev.Tags)
}

func StdTrace(exporter event.Exporter) event.Exporter {
	return func(ctx context.Context, ev core.Event, lm label.Map) context.Context {
		span := export.GetSpan(ctx)
		if span == nil {
			return exporter(ctx, ev, lm)
		}
		switch {
		case event.IsStart(ev):
			if span.ParentID.IsValid() {
				region := trace.StartRegion(ctx, span.Name)
				ctx = context.WithValue(ctx, traceKey, region)
			} else {
				var task *trace.Task
				ctx, task = trace.NewTask(ctx, span.Name)
				ctx = context.WithValue(ctx, traceKey, task)
			}
			// Log the start event as it may contain useful labels.
			msg := formatEvent(ctx, ev, lm)
			trace.Log(ctx, "start", msg)
		case event.IsLog(ev):
			category := ""
			if event.IsError(ev) {
				category = "error"
			}
			msg := formatEvent(ctx, ev, lm)
			trace.Log(ctx, category, msg)
		case event.IsEnd(ev):
			if v := ctx.Value(traceKey); v != nil {
				v.(interface{ End() }).End()
			}
		}
		return exporter(ctx, ev, lm)
	}
}

func formatEvent(ctx context.Context, ev core.Event, lm label.Map) string {
	buf := &bytes.Buffer{}
	p := export.Printer{}
	p.WriteEvent(buf, ev, lm)
	return buf.String()
}

func (t *traces) ProcessEvent(ctx context.Context, ev core.Event, lm label.Map) context.Context {
	span := export.GetSpan(ctx)
	if span == nil {
		return ctx
	}

	switch {
	case event.IsStart(ev):
		// Just starting: add it to the unfinished map.
		// Allocate before the critical section.
		td := &traceSpan{
			TraceID:  span.ID.TraceID,
			SpanID:   span.ID.SpanID,
			ParentID: span.ParentID,
			Name:     span.Name,
			Start:    span.Start().At(),
			Tags:     renderLabels(span.Start()),
		}

		t.mu.Lock()
		defer t.mu.Unlock()

		t.addRecentLocked(td, true) // add start event

		if t.sets == nil {
			t.sets = make(map[string]*traceSet)
			t.unfinished = make(map[export.SpanContext]*traceSpan)
		}
		t.unfinished[span.ID] = td

		// Wire up parents if we have them.
		if span.ParentID.IsValid() {
			parentID := export.SpanContext{TraceID: span.ID.TraceID, SpanID: span.ParentID}
			if parent, ok := t.unfinished[parentID]; ok {
				td.parent = parent
				parent.ChildStartEnd = append(parent.ChildStartEnd, spanStartEnd{true, td})
			}
		}

	case event.IsEnd(ev):
		// Finishing: must be already in the map.
		// Allocate events before the critical section.
		events := span.Events()
		tdEvents := make([]traceEvent, len(events))
		for i, event := range events {
			tdEvents[i] = traceEvent{
				Time: event.At(),
				Tags: renderLabels(event),
			}
		}

		t.mu.Lock()
		defer t.mu.Unlock()
		td, found := t.unfinished[span.ID]
		if !found {
			return ctx // if this happens we are in a bad place
		}
		delete(t.unfinished, span.ID)
		td.Finish = span.Finish().At()
		td.Duration = span.Finish().At().Sub(span.Start().At())
		td.Events = tdEvents
		t.addRecentLocked(td, false) // add end event

		set, ok := t.sets[span.Name]
		if !ok {
			set = &traceSet{Name: span.Name}
			t.sets[span.Name] = set
		}
		set.Last = td
		if set.Longest == nil || set.Last.Duration > set.Longest.Duration {
			set.Longest = set.Last
		}
		if td.parent != nil {
			td.parent.ChildStartEnd = append(td.parent.ChildStartEnd, spanStartEnd{false, td})
		} else {
			fillOffsets(td, td.Start)
		}
	}
	return ctx
}

// addRecentLocked appends a start or end event to the "recent" log,
// evicting an old entry if necessary.
func (t *traces) addRecentLocked(span *traceSpan, start bool) {
	t.recent = append(t.recent, spanStartEnd{Start: start, Span: span})

	const maxRecent = 100 // number of log entries before eviction
	for len(t.recent) > maxRecent {
		t.recent[0] = spanStartEnd{} // aid GC
		t.recent = t.recent[1:]
		t.recentEvictions++

		// Using a slice as a FIFO queue leads to unbounded growth
		// as Go's GC cannot collect the ever-growing unused prefix.
		// So, compact it periodically.
		if t.recentEvictions%maxRecent == 0 {
			t.recent = append([]spanStartEnd(nil), t.recent...)
		}
	}
}

// getData returns the TraceResults rendered by TraceTmpl for the /trace[/name] endpoint.
func (t *traces) getData(req *http.Request) interface{} {
	// TODO(adonovan): the HTTP request doesn't acquire the mutex
	// for t or for each span! Audit and fix.

	// Sort last/longest sets by name.
	traces := make([]*traceSet, 0, len(t.sets))
	for _, set := range t.sets {
		traces = append(traces, set)
	}
	sort.Slice(traces, func(i, j int) bool {
		return traces[i].Name < traces[j].Name
	})

	return TraceResults{
		Traces:   traces,
		Selected: t.sets[strings.TrimPrefix(req.URL.Path, "/trace/")], // may be nil
		Recent:   t.recent,
	}
}

func fillOffsets(td *traceSpan, start time.Time) {
	for i := range td.Events {
		td.Events[i].Offset = td.Events[i].Time.Sub(start)
	}
	for _, child := range td.ChildStartEnd {
		if !child.Start {
			fillOffsets(child.Span, start)
		}
	}
}

func renderLabels(labels label.List) string {
	buf := &bytes.Buffer{}
	for index := 0; labels.Valid(index); index++ {
		// The 'start' label duplicates the span name, so discard it.
		if l := labels.Label(index); l.Valid() && l.Key().Name() != "start" {
			fmt.Fprintf(buf, "%v ", l)
		}
	}
	return buf.String()
}

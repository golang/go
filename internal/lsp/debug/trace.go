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
	"sort"
	"strings"
	"sync"
	"time"

	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/event/core"
	"golang.org/x/tools/internal/event/export"
	"golang.org/x/tools/internal/event/label"
)

var traceTmpl = template.Must(template.Must(baseTemplate.Clone()).Parse(`
{{define "title"}}Trace Information{{end}}
{{define "body"}}
	{{range .Traces}}<a href="/trace/{{.Name}}">{{.Name}}</a> last: {{.Last.Duration}}, longest: {{.Longest.Duration}}<br>{{end}}
	{{if .Selected}}
		<H2>{{.Selected.Name}}</H2>
		{{if .Selected.Last}}<H3>Last</H3><ul>{{template "details" .Selected.Last}}</ul>{{end}}
		{{if .Selected.Longest}}<H3>Longest</H3><ul>{{template "details" .Selected.Longest}}</ul>{{end}}
	{{end}}
{{end}}
{{define "details"}}
	<li>{{.Offset}} {{.Name}} {{.Duration}} {{.Tags}}</li>
	{{if .Events}}<ul class=events>{{range .Events}}<li>{{.Offset}} {{.Tags}}</li>{{end}}</ul>{{end}}
	{{if .Children}}<ul>{{range .Children}}{{template "details" .}}{{end}}</ul>{{end}}
{{end}}
`))

type traces struct {
	mu         sync.Mutex
	sets       map[string]*traceSet
	unfinished map[export.SpanContext]*traceData
}

type traceResults struct {
	Traces   []*traceSet
	Selected *traceSet
}

type traceSet struct {
	Name    string
	Last    *traceData
	Longest *traceData
}

type traceData struct {
	TraceID  export.TraceID
	SpanID   export.SpanID
	ParentID export.SpanID
	Name     string
	Start    time.Time
	Finish   time.Time
	Offset   time.Duration
	Duration time.Duration
	Tags     string
	Events   []traceEvent
	Children []*traceData
}

type traceEvent struct {
	Time   time.Time
	Offset time.Duration
	Tags   string
}

func (t *traces) ProcessEvent(ctx context.Context, ev core.Event, lm label.Map) context.Context {
	t.mu.Lock()
	defer t.mu.Unlock()
	span := export.GetSpan(ctx)
	if span == nil {
		return ctx
	}

	switch {
	case event.IsStart(ev):
		if t.sets == nil {
			t.sets = make(map[string]*traceSet)
			t.unfinished = make(map[export.SpanContext]*traceData)
		}
		// just starting, add it to the unfinished map
		td := &traceData{
			TraceID:  span.ID.TraceID,
			SpanID:   span.ID.SpanID,
			ParentID: span.ParentID,
			Name:     span.Name,
			Start:    span.Start().At(),
			Tags:     renderLabels(span.Start()),
		}
		t.unfinished[span.ID] = td
		// and wire up parents if we have them
		if !span.ParentID.IsValid() {
			return ctx
		}
		parentID := export.SpanContext{TraceID: span.ID.TraceID, SpanID: span.ParentID}
		parent, found := t.unfinished[parentID]
		if !found {
			// trace had an invalid parent, so it cannot itself be valid
			return ctx
		}
		parent.Children = append(parent.Children, td)

	case event.IsEnd(ev):
		// finishing, must be already in the map
		td, found := t.unfinished[span.ID]
		if !found {
			return ctx // if this happens we are in a bad place
		}
		delete(t.unfinished, span.ID)

		td.Finish = span.Finish().At()
		td.Duration = span.Finish().At().Sub(span.Start().At())
		events := span.Events()
		td.Events = make([]traceEvent, len(events))
		for i, event := range events {
			td.Events[i] = traceEvent{
				Time: event.At(),
				Tags: renderLabels(event),
			}
		}

		set, ok := t.sets[span.Name]
		if !ok {
			set = &traceSet{Name: span.Name}
			t.sets[span.Name] = set
		}
		set.Last = td
		if set.Longest == nil || set.Last.Duration > set.Longest.Duration {
			set.Longest = set.Last
		}
		if !td.ParentID.IsValid() {
			fillOffsets(td, td.Start)
		}
	}
	return ctx
}

func (t *traces) getData(req *http.Request) interface{} {
	if len(t.sets) == 0 {
		return nil
	}
	data := traceResults{}
	data.Traces = make([]*traceSet, 0, len(t.sets))
	for _, set := range t.sets {
		data.Traces = append(data.Traces, set)
	}
	sort.Slice(data.Traces, func(i, j int) bool { return data.Traces[i].Name < data.Traces[j].Name })
	if bits := strings.SplitN(req.URL.Path, "/trace/", 2); len(bits) > 1 {
		data.Selected = t.sets[bits[1]]
	}
	return data
}

func fillOffsets(td *traceData, start time.Time) {
	td.Offset = td.Start.Sub(start)
	for i := range td.Events {
		td.Events[i].Offset = td.Events[i].Time.Sub(start)
	}
	for _, child := range td.Children {
		fillOffsets(child, start)
	}
}

func renderLabels(labels label.List) string {
	buf := &bytes.Buffer{}
	for index := 0; labels.Valid(index); index++ {
		if l := labels.Label(index); l.Valid() {
			fmt.Fprintf(buf, "%v ", l)
		}
	}
	return buf.String()
}

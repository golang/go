// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package debug

import (
	"bytes"
	"fmt"
	"html/template"
	"net/http"
	"sort"
	"strings"
	"time"

	"golang.org/x/tools/internal/lsp/telemetry/tag"
	"golang.org/x/tools/internal/lsp/telemetry/trace"
)

var traceTmpl = template.Must(template.Must(BaseTemplate.Clone()).Parse(`
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
	sets       map[string]*traceSet
	unfinished map[trace.SpanID]*traceData
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
	ID       trace.SpanID
	ParentID trace.SpanID
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

func (t *traces) export(span *trace.Span) {
	if t.sets == nil {
		t.sets = make(map[string]*traceSet)
		t.unfinished = make(map[trace.SpanID]*traceData)
	}
	// is this a completed span?
	if span.Finish.IsZero() {
		t.start(span)
	} else {
		t.finish(span)
	}
}

func (t *traces) start(span *trace.Span) {
	// just starting, add it to the unfinished map
	td := &traceData{
		ID:       span.SpanID,
		ParentID: span.ParentID,
		Name:     span.Name,
		Start:    span.Start,
		Tags:     renderTags(span.Tags),
	}
	t.unfinished[span.SpanID] = td
	// and wire up parents if we have them
	if !span.ParentID.IsValid() {
		return
	}
	parent, found := t.unfinished[span.ParentID]
	if !found {
		// trace had an invalid parent, so it cannot itself be valid
		return
	}
	parent.Children = append(parent.Children, td)

}

func (t *traces) finish(span *trace.Span) {
	// finishing, must be already in the map
	td, found := t.unfinished[span.SpanID]
	if !found {
		return // if this happens we are in a bad place
	}
	delete(t.unfinished, span.SpanID)

	td.Finish = span.Finish
	td.Duration = span.Finish.Sub(span.Start)
	td.Events = make([]traceEvent, len(span.Events))
	for i, event := range span.Events {
		td.Events[i] = traceEvent{
			Time: event.Time,
			Tags: renderTags(event.Tags),
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

func renderTags(tags tag.List) string {
	buf := &bytes.Buffer{}
	for _, tag := range tags {
		fmt.Fprintf(buf, "%v=%q ", tag.Key, tag.Value)
	}
	return buf.String()
}

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"cmp"
	"fmt"
	"html/template"
	"internal/trace"
	"internal/trace/traceviewer"
	tracev2 "internal/trace/v2"
	"log"
	"net/http"
	"slices"
	"strings"
	"time"
)

// UserTasksHandlerFunc returns a HandlerFunc that reports all tasks found in the trace.
func UserTasksHandlerFunc(t *parsedTrace) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		tasks := t.summary.Tasks

		// Summarize groups of tasks with the same name.
		summary := make(map[string]taskStats)
		for _, task := range tasks {
			stats, ok := summary[task.Name]
			if !ok {
				stats.Type = task.Name
			}
			stats.add(task)
			summary[task.Name] = stats
		}

		// Sort tasks by type.
		userTasks := make([]taskStats, 0, len(summary))
		for _, stats := range summary {
			userTasks = append(userTasks, stats)
		}
		slices.SortFunc(userTasks, func(a, b taskStats) int {
			return cmp.Compare(a.Type, b.Type)
		})

		// Emit table.
		err := templUserTaskTypes.Execute(w, userTasks)
		if err != nil {
			http.Error(w, fmt.Sprintf("failed to execute template: %v", err), http.StatusInternalServerError)
			return
		}
	}
}

type taskStats struct {
	Type      string
	Count     int                       // Complete + incomplete tasks
	Histogram traceviewer.TimeHistogram // Complete tasks only
}

func (s *taskStats) UserTaskURL(complete bool) func(min, max time.Duration) string {
	return func(min, max time.Duration) string {
		return fmt.Sprintf("/usertask?type=%s&complete=%v&latmin=%v&latmax=%v", template.URLQueryEscaper(s.Type), template.URLQueryEscaper(complete), template.URLQueryEscaper(min), template.URLQueryEscaper(max))
	}
}

func (s *taskStats) add(task *trace.UserTaskSummary) {
	s.Count++
	if task.Complete() {
		s.Histogram.Add(task.End.Time().Sub(task.Start.Time()))
	}
}

var templUserTaskTypes = template.Must(template.New("").Parse(`
<!DOCTYPE html>
<title>Tasks</title>
<style>` + traceviewer.CommonStyle + `
.histoTime {
  width: 20%;
  white-space:nowrap;
}
th {
  background-color: #050505;
  color: #fff;
}
table {
  border-collapse: collapse;
}
td,
th {
  padding-left: 8px;
  padding-right: 8px;
  padding-top: 4px;
  padding-bottom: 4px;
}
</style>
<body>
Search log text: <form action="/usertask"><input name="logtext" type="text"><input type="submit"></form><br>
<table border="1" sortable="1">
<tr>
<th>Task type</th>
<th>Count</th>
<th>Duration distribution (complete tasks)</th>
</tr>
{{range $}}
  <tr>
    <td>{{.Type}}</td>
    <td><a href="/usertask?type={{.Type}}">{{.Count}}</a></td>
    <td>{{.Histogram.ToHTML (.UserTaskURL true)}}</td>
  </tr>
{{end}}
</table>
</body>
</html>
`))

// UserTaskHandlerFunc returns a HandlerFunc that presents the details of the selected tasks.
func UserTaskHandlerFunc(t *parsedTrace) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		filter, err := newTaskFilter(r)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		type event struct {
			WhenString string
			Elapsed    time.Duration
			Goroutine  tracev2.GoID
			What       string
			// TODO: include stack trace of creation time
		}
		type task struct {
			WhenString string
			ID         tracev2.TaskID
			Duration   time.Duration
			Complete   bool
			Events     []event
			Start, End time.Duration // Time since the beginning of the trace
			GCTime     time.Duration
		}
		var tasks []task
		for _, summary := range t.summary.Tasks {
			if !filter.match(t, summary) {
				continue
			}

			// Collect all the events for the task.
			var rawEvents []*tracev2.Event
			if summary.Start != nil {
				rawEvents = append(rawEvents, summary.Start)
			}
			if summary.End != nil {
				rawEvents = append(rawEvents, summary.End)
			}
			rawEvents = append(rawEvents, summary.Logs...)
			for _, r := range summary.Regions {
				if r.Start != nil {
					rawEvents = append(rawEvents, r.Start)
				}
				if r.End != nil {
					rawEvents = append(rawEvents, r.End)
				}
			}

			// Sort them.
			slices.SortStableFunc(rawEvents, func(a, b *tracev2.Event) int {
				return cmp.Compare(a.Time(), b.Time())
			})

			// Summarize them.
			var events []event
			last := t.startTime()
			for _, ev := range rawEvents {
				what := describeEvent(ev)
				if what == "" {
					continue
				}
				sinceStart := ev.Time().Sub(t.startTime())
				events = append(events, event{
					WhenString: fmt.Sprintf("%2.9f", sinceStart.Seconds()),
					Elapsed:    ev.Time().Sub(last),
					What:       what,
					Goroutine:  primaryGoroutine(ev),
				})
				last = ev.Time()
			}
			taskSpan := taskInterval(t, summary)
			taskStart := taskSpan.start.Sub(t.startTime())

			// Produce the task summary.
			tasks = append(tasks, task{
				WhenString: fmt.Sprintf("%2.9fs", taskStart.Seconds()),
				Duration:   taskSpan.duration(),
				ID:         summary.ID,
				Complete:   summary.Complete(),
				Events:     events,
				Start:      taskStart,
				End:        taskStart + taskSpan.duration(),
			})
		}
		// Sort the tasks by duration.
		slices.SortFunc(tasks, func(a, b task) int {
			return cmp.Compare(a.Duration, b.Duration)
		})

		// Emit table.
		err = templUserTaskType.Execute(w, struct {
			Name  string
			Tasks []task
		}{
			Name:  filter.name,
			Tasks: tasks,
		})
		if err != nil {
			log.Printf("failed to execute template: %v", err)
			http.Error(w, fmt.Sprintf("failed to execute template: %v", err), http.StatusInternalServerError)
			return
		}
	}
}

var templUserTaskType = template.Must(template.New("userTask").Funcs(template.FuncMap{
	"elapsed":       elapsed,
	"asMillisecond": asMillisecond,
	"trimSpace":     strings.TrimSpace,
}).Parse(`
<!DOCTYPE html>
<title>Tasks: {{.Name}}</title>
<style>` + traceviewer.CommonStyle + `
body {
  font-family: sans-serif;
}
table#req-status td.family {
  padding-right: 2em;
}
table#req-status td.active {
  padding-right: 1em;
}
table#req-status td.empty {
  color: #aaa;
}
table#reqs {
  margin-top: 1em;
  border-collapse: collapse;
}
table#reqs tr.first {
  font-weight: bold;
}
table#reqs td {
  font-family: monospace;
}
table#reqs td.when {
  text-align: right;
  white-space: nowrap;
}
table#reqs td.elapsed {
  padding: 0 0.5em;
  text-align: right;
  white-space: pre;
  width: 10em;
}
address {
  font-size: smaller;
  margin-top: 5em;
}
</style>
<body>

<h2>User Task: {{.Name}}</h2>

Search log text: <form onsubmit="window.location.search+='&logtext='+window.logtextinput.value; return false">
<input name="logtext" id="logtextinput" type="text"><input type="submit">
</form><br>

<table id="reqs">
	<tr>
		<th>When</th>
		<th>Elapsed</th>
		<th>Goroutine</th>
		<th>Events</th>
	</tr>
	{{range $el := $.Tasks}}
	<tr class="first">
		<td class="when">{{$el.WhenString}}</td>
		<td class="elapsed">{{$el.Duration}}</td>
		<td></td>
		<td>
			<a href="/trace?focustask={{$el.ID}}#{{asMillisecond $el.Start}}:{{asMillisecond $el.End}}">Task {{$el.ID}}</a>
			<a href="/trace?taskid={{$el.ID}}#{{asMillisecond $el.Start}}:{{asMillisecond $el.End}}">(goroutine view)</a>
			({{if .Complete}}complete{{else}}incomplete{{end}})
		</td>
	</tr>
	{{range $el.Events}}
	<tr>
		<td class="when">{{.WhenString}}</td>
		<td class="elapsed">{{elapsed .Elapsed}}</td>
		<td class="goid">{{.Goroutine}}</td>
		<td>{{.What}}</td>
	</tr>
	{{end}}
    {{end}}
</body>
</html>
`))

// taskFilter represents a task filter specified by a user of cmd/trace.
type taskFilter struct {
	name string
	cond []func(*parsedTrace, *trace.UserTaskSummary) bool
}

// match returns true if a task, described by its ID and summary, matches
// the filter.
func (f *taskFilter) match(t *parsedTrace, task *trace.UserTaskSummary) bool {
	if t == nil {
		return false
	}
	for _, c := range f.cond {
		if !c(t, task) {
			return false
		}
	}
	return true
}

// newTaskFilter creates a new task filter from URL query variables.
func newTaskFilter(r *http.Request) (*taskFilter, error) {
	if err := r.ParseForm(); err != nil {
		return nil, err
	}

	var name []string
	var conditions []func(*parsedTrace, *trace.UserTaskSummary) bool

	param := r.Form
	if typ, ok := param["type"]; ok && len(typ) > 0 {
		name = append(name, fmt.Sprintf("%q", typ[0]))
		conditions = append(conditions, func(_ *parsedTrace, task *trace.UserTaskSummary) bool {
			return task.Name == typ[0]
		})
	}
	if complete := r.FormValue("complete"); complete == "1" {
		name = append(name, "complete")
		conditions = append(conditions, func(_ *parsedTrace, task *trace.UserTaskSummary) bool {
			return task.Complete()
		})
	} else if complete == "0" {
		name = append(name, "incomplete")
		conditions = append(conditions, func(_ *parsedTrace, task *trace.UserTaskSummary) bool {
			return !task.Complete()
		})
	}
	if lat, err := time.ParseDuration(r.FormValue("latmin")); err == nil {
		name = append(name, fmt.Sprintf("latency >= %s", lat))
		conditions = append(conditions, func(t *parsedTrace, task *trace.UserTaskSummary) bool {
			return task.Complete() && taskInterval(t, task).duration() >= lat
		})
	}
	if lat, err := time.ParseDuration(r.FormValue("latmax")); err == nil {
		name = append(name, fmt.Sprintf("latency <= %s", lat))
		conditions = append(conditions, func(t *parsedTrace, task *trace.UserTaskSummary) bool {
			return task.Complete() && taskInterval(t, task).duration() <= lat
		})
	}
	if text := r.FormValue("logtext"); text != "" {
		name = append(name, fmt.Sprintf("log contains %q", text))
		conditions = append(conditions, func(_ *parsedTrace, task *trace.UserTaskSummary) bool {
			return taskMatches(task, text)
		})
	}

	return &taskFilter{name: strings.Join(name, ","), cond: conditions}, nil
}

func taskInterval(t *parsedTrace, s *trace.UserTaskSummary) interval {
	var i interval
	if s.Start != nil {
		i.start = s.Start.Time()
	} else {
		i.start = t.startTime()
	}
	if s.End != nil {
		i.end = s.End.Time()
	} else {
		i.end = t.endTime()
	}
	return i
}

func taskMatches(t *trace.UserTaskSummary, text string) bool {
	matches := func(s string) bool {
		return strings.Contains(s, text)
	}
	if matches(t.Name) {
		return true
	}
	for _, r := range t.Regions {
		if matches(r.Name) {
			return true
		}
	}
	for _, ev := range t.Logs {
		log := ev.Log()
		if matches(log.Category) {
			return true
		}
		if matches(log.Message) {
			return true
		}
	}
	return false
}

func describeEvent(ev *tracev2.Event) string {
	switch ev.Kind() {
	case tracev2.EventStateTransition:
		st := ev.StateTransition()
		if st.Resource.Kind != tracev2.ResourceGoroutine {
			return ""
		}
		old, new := st.Goroutine()
		return fmt.Sprintf("%s -> %s", old, new)
	case tracev2.EventRegionBegin:
		return fmt.Sprintf("region %q begin", ev.Region().Type)
	case tracev2.EventRegionEnd:
		return fmt.Sprintf("region %q end", ev.Region().Type)
	case tracev2.EventTaskBegin:
		t := ev.Task()
		return fmt.Sprintf("task %q (D %d, parent %d) begin", t.Type, t.ID, t.Parent)
	case tracev2.EventTaskEnd:
		return "task end"
	case tracev2.EventLog:
		log := ev.Log()
		if log.Category != "" {
			return fmt.Sprintf("log %q", log.Message)
		}
		return fmt.Sprintf("log (category: %s): %q", log.Category, log.Message)
	}
	return ""
}

func primaryGoroutine(ev *tracev2.Event) tracev2.GoID {
	if ev.Kind() != tracev2.EventStateTransition {
		return ev.Goroutine()
	}
	st := ev.StateTransition()
	if st.Resource.Kind != tracev2.ResourceGoroutine {
		return tracev2.NoGoroutine
	}
	return st.Resource.Goroutine()
}

func elapsed(d time.Duration) string {
	b := fmt.Appendf(nil, "%.9f", d.Seconds())

	// For subsecond durations, blank all zeros before decimal point,
	// and all zeros between the decimal point and the first non-zero digit.
	if d < time.Second {
		dot := bytes.IndexByte(b, '.')
		for i := 0; i < dot; i++ {
			b[i] = ' '
		}
		for i := dot + 1; i < len(b); i++ {
			if b[i] == '0' {
				b[i] = ' '
			} else {
				break
			}
		}
	}
	return string(b)
}

func asMillisecond(d time.Duration) float64 {
	return float64(d.Nanoseconds()) / float64(time.Millisecond)
}

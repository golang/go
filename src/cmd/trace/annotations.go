package main

import (
	"bytes"
	"fmt"
	"html/template"
	"internal/trace"
	"log"
	"math"
	"net/http"
	"sort"
	"strings"
	"time"
)

func init() {
	http.HandleFunc("/usertasks", httpUserTasks)
	http.HandleFunc("/usertask", httpUserTask)
}

// httpUserTasks reports all tasks found in the trace.
func httpUserTasks(w http.ResponseWriter, r *http.Request) {
	tasks, err := analyzeAnnotations()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	summary := make(map[string]taskStats)
	for _, task := range tasks {
		stats, ok := summary[task.name]
		if !ok {
			stats.Type = task.name
		}

		stats.add(task)
		summary[task.name] = stats
	}

	// Sort tasks by type.
	userTasks := make([]taskStats, 0, len(summary))
	for _, stats := range summary {
		userTasks = append(userTasks, stats)
	}
	sort.Slice(userTasks, func(i, j int) bool {
		return userTasks[i].Type < userTasks[j].Type
	})

	// Emit table.
	err = templUserTaskTypes.Execute(w, userTasks)
	if err != nil {
		http.Error(w, fmt.Sprintf("failed to execute template: %v", err), http.StatusInternalServerError)
		return
	}
}

// httpUserTask presents the details of the selected tasks.
func httpUserTask(w http.ResponseWriter, r *http.Request) {
	filter, err := newTaskFilter(r)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	tasks, err := analyzeAnnotations()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	type event struct {
		WhenString string
		Elapsed    time.Duration
		Go         uint64
		What       string
		// TODO: include stack trace of creation time
	}
	type entry struct {
		WhenString string
		ID         uint64
		Duration   time.Duration
		Complete   bool
		Events     []event
		Start, End time.Duration // Time since the beginning of the trace
	}

	base := time.Duration(firstTimestamp()) * time.Nanosecond // trace start

	var data []entry

	for _, task := range tasks {
		if !filter.match(task) {
			continue
		}
		var events []event
		var last time.Duration

		for i, ev := range task.events {
			when := time.Duration(ev.Ts)*time.Nanosecond - base
			elapsed := time.Duration(ev.Ts)*time.Nanosecond - last
			if i == 0 {
				elapsed = 0
			}

			events = append(events, event{
				WhenString: fmt.Sprintf("%2.9f", when.Seconds()),
				Elapsed:    elapsed,
				What:       describeEvent(ev),
				Go:         ev.G,
			})
			last = time.Duration(ev.Ts) * time.Nanosecond
		}
		data = append(data, entry{
			WhenString: fmt.Sprintf("%2.9fs", (time.Duration(task.firstTimestamp())*time.Nanosecond - base).Seconds()),
			Duration:   task.duration(),
			ID:         task.id,
			Complete:   task.complete(),
			Events:     events,
			Start:      time.Duration(task.firstTimestamp()) * time.Nanosecond,
			End:        time.Duration(task.lastTimestamp()) * time.Nanosecond,
		})
	}
	sort.Slice(data, func(i, j int) bool {
		return data[i].Duration < data[j].Duration
	})

	// Emit table.
	err = templUserTaskType.Execute(w, struct {
		Name  string
		Entry []entry
	}{
		Name:  filter.name,
		Entry: data,
	})
	if err != nil {
		log.Printf("failed to execute template: %v", err)
		http.Error(w, fmt.Sprintf("failed to execute template: %v", err), http.StatusInternalServerError)
		return
	}
}

// analyzeAnnotations analyzes user annotation events and
// returns the task descriptors keyed by internal task id.
func analyzeAnnotations() (map[uint64]*taskDesc, error) {
	res, err := parseTrace()
	if err != nil {
		return nil, fmt.Errorf("failed to parse trace: %v", err)
	}

	events := res.Events
	if len(events) == 0 {
		return nil, fmt.Errorf("empty trace")
	}

	tasks := allTasks{}
	activeSpans := map[uint64][]*trace.Event{} // goid to active span start events

	for _, ev := range events {
		goid := ev.G

		switch typ := ev.Type; typ {
		case trace.EvUserTaskCreate, trace.EvUserTaskEnd, trace.EvUserLog, trace.EvUserSpan:
			taskid := ev.Args[0]
			task := tasks.task(taskid)
			task.addEvent(ev)

			if typ == trace.EvUserSpan {
				mode := ev.Args[1]
				spans := activeSpans[goid]
				if mode == 0 { // start
					activeSpans[goid] = append(spans, ev) // push
				} else { // end
					if n := len(spans); n > 1 {
						activeSpans[goid] = spans[:n-1] // pop
					} else if n == 1 {
						delete(activeSpans, goid)
					}
				}
			}

		case trace.EvGoCreate:
			// When a goroutine is newly created, it inherits the task
			// of the active span if any.
			//
			// TODO(hyangah): the task info needs to propagate
			// to all decendents, not only to the immediate child.
			spans := activeSpans[goid]
			if len(spans) == 0 {
				continue
			}
			taskid := spans[len(spans)-1].Args[0]
			task := tasks.task(taskid)
			task.addEvent(ev)
		}
	}
	return tasks, nil
}

// taskDesc represents a task.
type taskDesc struct {
	name       string // user-provided task name
	id         uint64 // internal task id
	events     []*trace.Event
	spans      []*spanDesc               // associated spans
	goroutines map[uint64][]*trace.Event // Events grouped by goroutine id

	create *trace.Event // Task create event
	end    *trace.Event // Task end event
}

func (task *taskDesc) String() string {
	if task == nil {
		return "task <nil>"
	}
	wb := new(bytes.Buffer)
	fmt.Fprintf(wb, "task %d:\t%s\n", task.id, task.name)
	fmt.Fprintf(wb, "\tstart: %v end: %v complete: %t\n", task.firstTimestamp(), task.lastTimestamp(), task.complete())
	fmt.Fprintf(wb, "\t%d goroutines\n", len(task.goroutines))
	fmt.Fprintf(wb, "\t%d spans:\n", len(task.spans))
	for _, s := range task.spans {
		fmt.Fprintf(wb, "\t\t%s(goid=%d)\n", s.name, s.goid)
	}
	return wb.String()
}

// spanDesc represents a span.
type spanDesc struct {
	name  string       // user-provided span name
	task  *taskDesc    // can be nil
	goid  uint64       // id of goroutine where the span was defined
	start *trace.Event // span start event
	end   *trace.Event // span end event (user span end, goroutine end)
}

type allTasks map[uint64]*taskDesc

func (tasks allTasks) task(taskID uint64) *taskDesc {
	if taskID == 0 {
		return nil // notask
	}

	t, ok := tasks[taskID]
	if ok {
		return t
	}

	t = &taskDesc{
		id:         taskID,
		goroutines: make(map[uint64][]*trace.Event),
	}
	tasks[taskID] = t
	return t
}

func (task *taskDesc) addEvent(ev *trace.Event) {
	if task == nil {
		// TODO(hyangah): handle spans with no task.
		return
	}

	if ev != task.lastEvent() {
		goid := ev.G
		task.events = append(task.events, ev)
		task.goroutines[goid] = append(task.goroutines[goid], ev)
	}

	switch typ := ev.Type; typ {
	case trace.EvUserTaskCreate:
		task.name = ev.SArgs[0]
		task.create = ev
	case trace.EvUserTaskEnd:
		task.end = ev
	case trace.EvUserSpan:
		if mode := ev.Args[1]; mode == 0 { // start
			task.spans = append(task.spans, &spanDesc{
				name:  ev.SArgs[0],
				task:  task,
				goid:  ev.G,
				start: ev,
				end:   ev.Link})
		} else if ev.Link == nil { // span end without matching start
			task.spans = append(task.spans, &spanDesc{
				name: ev.SArgs[0],
				task: task,
				goid: ev.G,
				end:  ev})
		}
	}
}

// complete is true only if both start and end events of this task
// are present in the trace.
func (task *taskDesc) complete() bool {
	if task == nil {
		return false
	}
	return task.create != nil && task.end != nil
}

// firstTimestamp returns the first timestamp of this task found in
// this trace. If the trace does not contain the task creation event,
// the first timestamp of the trace will be returned.
func (task *taskDesc) firstTimestamp() int64 {
	if task != nil && task.create != nil {
		return task.create.Ts
	}
	return firstTimestamp()
}

// lastTimestamp returns the last timestamp of this task in this
// trace. If the trace does not contain the task end event, the last
// timestamp of the trace will be returned.
func (task *taskDesc) lastTimestamp() int64 {
	if task != nil && task.end != nil {
		return task.end.Ts
	}
	return lastTimestamp()
}

func (task *taskDesc) duration() time.Duration {
	return time.Duration(task.lastTimestamp()-task.firstTimestamp()) * time.Nanosecond
}

func (task *taskDesc) lastEvent() *trace.Event {
	if task == nil {
		return nil
	}

	if n := len(task.events); n > 0 {
		return task.events[n-1]
	}
	return nil
}

type taskFilter struct {
	name string
	cond []func(*taskDesc) bool
}

func (f *taskFilter) match(t *taskDesc) bool {
	if t == nil {
		return false
	}
	for _, c := range f.cond {
		if !c(t) {
			return false
		}
	}
	return true
}

func newTaskFilter(r *http.Request) (*taskFilter, error) {
	if err := r.ParseForm(); err != nil {
		return nil, err
	}

	var name []string
	var conditions []func(*taskDesc) bool

	param := r.Form
	if typ, ok := param["type"]; ok && len(typ) > 0 {
		name = append(name, "type="+typ[0])
		conditions = append(conditions, func(t *taskDesc) bool {
			return t.name == typ[0]
		})
	}
	if complete := r.FormValue("complete"); complete == "1" {
		name = append(name, "complete")
		conditions = append(conditions, func(t *taskDesc) bool {
			return t.complete()
		})
	} else if complete == "0" {
		name = append(name, "incomplete")
		conditions = append(conditions, func(t *taskDesc) bool {
			return !t.complete()
		})
	}
	if lat, err := time.ParseDuration(r.FormValue("latmin")); err == nil {
		name = append(name, fmt.Sprintf("latency >= %s", lat))
		conditions = append(conditions, func(t *taskDesc) bool {
			return t.complete() && t.duration() >= lat
		})
	}
	if lat, err := time.ParseDuration(r.FormValue("latmax")); err == nil {
		name = append(name, fmt.Sprintf("latency <= %s", lat))
		conditions = append(conditions, func(t *taskDesc) bool {
			return t.complete() && t.duration() <= lat
		})
	}

	return &taskFilter{name: strings.Join(name, ","), cond: conditions}, nil
}

type durationHistogram struct {
	Count                int
	Buckets              []int
	MinBucket, MaxBucket int
}

// Five buckets for every power of 10.
var logDiv = math.Log(math.Pow(10, 1.0/5))

func (h *durationHistogram) add(d time.Duration) {
	var bucket int
	if d > 0 {
		bucket = int(math.Log(float64(d)) / logDiv)
	}
	if len(h.Buckets) <= bucket {
		h.Buckets = append(h.Buckets, make([]int, bucket-len(h.Buckets)+1)...)
		h.Buckets = h.Buckets[:cap(h.Buckets)]
	}
	h.Buckets[bucket]++
	if bucket < h.MinBucket || h.MaxBucket == 0 {
		h.MinBucket = bucket
	}
	if bucket > h.MaxBucket {
		h.MaxBucket = bucket
	}
	h.Count++
}

func (h *durationHistogram) BucketMin(bucket int) time.Duration {
	return time.Duration(math.Exp(float64(bucket) * logDiv))
}

func niceDuration(d time.Duration) string {
	var rnd time.Duration
	var unit string
	switch {
	case d < 10*time.Microsecond:
		rnd, unit = time.Nanosecond, "ns"
	case d < 10*time.Millisecond:
		rnd, unit = time.Microsecond, "µs"
	case d < 10*time.Second:
		rnd, unit = time.Millisecond, "ms"
	default:
		rnd, unit = time.Second, "s "
	}
	return fmt.Sprintf("%d%s", d/rnd, unit)
}

func (h *durationHistogram) ToHTML(urlmaker func(min, max time.Duration) string) template.HTML {
	if h == nil || h.Count == 0 {
		return template.HTML("")
	}

	const barWidth = 400

	maxCount := 0
	for _, count := range h.Buckets {
		if count > maxCount {
			maxCount = count
		}
	}

	w := new(bytes.Buffer)
	fmt.Fprintf(w, `<table>`)
	for i := h.MinBucket; i <= h.MaxBucket; i++ {
		// Tick label.
		fmt.Fprintf(w, `<tr><td class="histoTime" align="right"><a href=%s>%s</a></td>`, urlmaker(h.BucketMin(i), h.BucketMin(i+1)), niceDuration(h.BucketMin(i)))
		// Bucket bar.
		width := h.Buckets[i] * barWidth / maxCount
		fmt.Fprintf(w, `<td><div style="width:%dpx;background:blue;top:.6em;position:relative">&nbsp;</div></td>`, width)
		// Bucket count.
		fmt.Fprintf(w, `<td align="right"><div style="top:.6em;position:relative">%d</div></td>`, h.Buckets[i])
		fmt.Fprintf(w, "</tr>\n")

	}
	// Final tick label.
	fmt.Fprintf(w, `<tr><td align="right">%s</td></tr>`, niceDuration(h.BucketMin(h.MaxBucket+1)))
	fmt.Fprintf(w, `</table>`)
	return template.HTML(w.String())
}

func (h *durationHistogram) String() string {
	const barWidth = 40

	labels := []string{}
	maxLabel := 0
	maxCount := 0
	for i := h.MinBucket; i <= h.MaxBucket; i++ {
		// TODO: This formatting is pretty awful.
		label := fmt.Sprintf("[%-12s%-11s)", h.BucketMin(i).String()+",", h.BucketMin(i+1))
		labels = append(labels, label)
		if len(label) > maxLabel {
			maxLabel = len(label)
		}
		count := h.Buckets[i]
		if count > maxCount {
			maxCount = count
		}
	}

	w := new(bytes.Buffer)
	for i := h.MinBucket; i <= h.MaxBucket; i++ {
		count := h.Buckets[i]
		bar := count * barWidth / maxCount
		fmt.Fprintf(w, "%*s %-*s %d\n", maxLabel, labels[i-h.MinBucket], barWidth, strings.Repeat("█", bar), count)
	}
	return w.String()
}

type taskStats struct {
	Type      string
	Count     int               // Complete + incomplete tasks
	Histogram durationHistogram // Complete tasks only
}

func (s *taskStats) UserTaskURL(complete bool) func(min, max time.Duration) string {
	return func(min, max time.Duration) string {
		return fmt.Sprintf("/usertask?type=%s&complete=%v&latmin=%v&latmax=%v", template.URLQueryEscaper(s.Type), template.URLQueryEscaper(complete), template.URLQueryEscaper(min), template.URLQueryEscaper(max))
	}
}

func (s *taskStats) add(task *taskDesc) {
	s.Count++
	if task.complete() {
		s.Histogram.add(task.duration())
	}
}

var templUserTaskTypes = template.Must(template.New("").Parse(`
<html>
<style type="text/css">
.histoTime {
   width: 20%;
   white-space:nowrap;
}

</style>
<body>
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

var templUserTaskType = template.Must(template.New("userTask").Funcs(template.FuncMap{
	"elapsed":       elapsed,
	"asMillisecond": asMillisecond,
	"trimSpace":     strings.TrimSpace,
}).Parse(`
<html>
<head> <title>User Task: {{.Name}} </title> </head>
        <style type="text/css">
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

<table id="reqs">
<tr><th>When</th><th>Elapsed</th><th>Goroutine ID</th><th>Events</th></tr>
     {{range $el := $.Entry}}
        <tr class="first">
                <td class="when">{{$el.WhenString}}</td>
                <td class="elapsed">{{$el.Duration}}</td>
		<td></td>
                <td><a href="/trace?taskid={{$el.ID}}#{{asMillisecond $el.Start}}:{{asMillisecond $el.End}}">Task {{$el.ID}}</a> ({{if .Complete}}complete{{else}}incomplete{{end}})</td>
        </tr>
        {{range $el.Events}}
        <tr>
                <td class="when">{{.WhenString}}</td>
                <td class="elapsed">{{elapsed .Elapsed}}</td>
		<td class="goid">{{.Go}}</td>
                <td>{{.What}}</td>
        </tr>
        {{end}}
    {{end}}
</body>
</html>
`))

func elapsed(d time.Duration) string {
	b := []byte(fmt.Sprintf("%.9f", d.Seconds()))

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
	return float64(d.Nanoseconds()) / 1e6
}

func describeEvent(ev *trace.Event) string {
	switch ev.Type {
	case trace.EvGoCreate:
		return fmt.Sprintf("created a new goroutine %d", ev.Args[0])
	case trace.EvGoEnd, trace.EvGoStop:
		return "goroutine stopped"
	case trace.EvUserLog:
		return fmt.Sprintf("%v=%v", ev.SArgs[0], ev.SArgs[1])
	case trace.EvUserSpan:
		if ev.Args[1] == 0 {
			duration := "unknown"
			if ev.Link != nil {
				duration = (time.Duration(ev.Link.Ts-ev.Ts) * time.Nanosecond).String()
			}
			return fmt.Sprintf("span %s started (duration: %v)", ev.SArgs[0], duration)
		}
		return fmt.Sprintf("span %s ended", ev.SArgs[0])
	case trace.EvUserTaskCreate:
		return fmt.Sprintf("task %v (id %d, parent %d) created", ev.SArgs[0], ev.Args[0], ev.Args[1])
		// TODO: add child task creation events into the parent task events
	case trace.EvUserTaskEnd:
		return "task end"
	}
	return ""
}

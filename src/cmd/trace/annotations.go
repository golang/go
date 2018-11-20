// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"fmt"
	"html/template"
	"log"
	"math"
	"net/http"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"time"

	trace "internal/traceparser"
)

func init() {
	http.HandleFunc("/usertasks", httpUserTasks)
	http.HandleFunc("/usertask", httpUserTask)
	http.HandleFunc("/userregions", httpUserRegions)
	http.HandleFunc("/userregion", httpUserRegion)
}

// httpUserTasks reports all tasks found in the trace.
func httpUserTasks(w http.ResponseWriter, r *http.Request) {
	res, err := analyzeAnnotations()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	tasks := res.tasks
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

func httpUserRegions(w http.ResponseWriter, r *http.Request) {
	res, err := analyzeAnnotations()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	allRegions := res.regions

	summary := make(map[regionTypeID]regionStats)
	for id, regions := range allRegions {
		stats, ok := summary[id]
		if !ok {
			stats.regionTypeID = id
		}
		for _, s := range regions {
			stats.add(s)
		}
		summary[id] = stats
	}
	// Sort regions by pc and name
	userRegions := make([]regionStats, 0, len(summary))
	for _, stats := range summary {
		userRegions = append(userRegions, stats)
	}
	sort.Slice(userRegions, func(i, j int) bool {
		if userRegions[i].Type != userRegions[j].Type {
			return userRegions[i].Type < userRegions[j].Type
		}
		return userRegions[i].Frame.PC < userRegions[j].Frame.PC
	})
	// Emit table.
	err = templUserRegionTypes.Execute(w, userRegions)
	if err != nil {
		http.Error(w, fmt.Sprintf("failed to execute template: %v", err), http.StatusInternalServerError)
		return
	}
}

func httpUserRegion(w http.ResponseWriter, r *http.Request) {
	filter, err := newRegionFilter(r)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	res, err := analyzeAnnotations()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	allRegions := res.regions

	var data []regionDesc

	var maxTotal int64
	for id, regions := range allRegions {
		for _, s := range regions {
			if !filter.match(id, s) {
				continue
			}
			data = append(data, s)
			if maxTotal < s.TotalTime {
				maxTotal = s.TotalTime
			}
		}
	}

	sortby := r.FormValue("sortby")
	_, ok := reflect.TypeOf(regionDesc{}).FieldByNameFunc(func(s string) bool {
		return s == sortby
	})
	if !ok {
		sortby = "TotalTime"
	}
	sort.Slice(data, func(i, j int) bool {
		ival := reflect.ValueOf(data[i]).FieldByName(sortby).Int()
		jval := reflect.ValueOf(data[j]).FieldByName(sortby).Int()
		return ival > jval
	})

	err = templUserRegionType.Execute(w, struct {
		MaxTotal int64
		Data     []regionDesc
		Name     string
	}{
		MaxTotal: maxTotal,
		Data:     data,
		Name:     filter.name,
	})
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

	res, err := analyzeAnnotations()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	tasks := res.tasks

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
		GCTime     time.Duration
	}

	base := time.Duration(firstTimestamp()) * time.Nanosecond // trace start

	var data []entry

	for _, task := range tasks {
		if !filter.match(task) {
			continue
		}
		// merge events in the task.events and task.regions.Start
		rawEvents := append([]*trace.Event{}, task.events...)
		for _, s := range task.regions {
			if s.Start != nil {
				rawEvents = append(rawEvents, s.Start)
			}
		}
		sort.SliceStable(rawEvents, func(i, j int) bool { return rawEvents[i].Ts < rawEvents[j].Ts })

		var events []event
		var last time.Duration
		for i, ev := range rawEvents {
			when := time.Duration(ev.Ts)*time.Nanosecond - base
			elapsed := time.Duration(ev.Ts)*time.Nanosecond - last
			if i == 0 {
				elapsed = 0
			}

			what := describeEvent(ev)
			if what != "" {
				events = append(events, event{
					WhenString: fmt.Sprintf("%2.9f", when.Seconds()),
					Elapsed:    elapsed,
					What:       what,
					Go:         ev.G,
				})
				last = time.Duration(ev.Ts) * time.Nanosecond
			}
		}

		data = append(data, entry{
			WhenString: fmt.Sprintf("%2.9fs", (time.Duration(task.firstTimestamp())*time.Nanosecond - base).Seconds()),
			Duration:   task.duration(),
			ID:         task.id,
			Complete:   task.complete(),
			Events:     events,
			Start:      time.Duration(task.firstTimestamp()) * time.Nanosecond,
			End:        time.Duration(task.endTimestamp()) * time.Nanosecond,
			GCTime:     task.overlappingGCDuration(res.gcEvents),
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

type annotationAnalysisResult struct {
	tasks    map[uint64]*taskDesc          // tasks
	regions  map[regionTypeID][]regionDesc // regions
	gcEvents []*trace.Event                // GCStartevents, sorted
}

type regionTypeID struct {
	Frame trace.Frame // top frame
	Type  string
}

// analyzeAnnotations analyzes user annotation events and
// returns the task descriptors keyed by internal task id.
func analyzeAnnotations() (annotationAnalysisResult, error) {
	res, err := parseTrace()
	if err != nil {
		return annotationAnalysisResult{}, fmt.Errorf("failed to parse trace: %v", err)
	}

	events := res.Events
	if len(events) == 0 {
		return annotationAnalysisResult{}, fmt.Errorf("empty trace")
	}

	tasks := allTasks{}
	regions := map[regionTypeID][]regionDesc{}
	var gcEvents []*trace.Event

	for _, ev := range events {
		switch typ := ev.Type; typ {
		case trace.EvUserTaskCreate, trace.EvUserTaskEnd, trace.EvUserLog:
			taskid := ev.Args[0]
			task := tasks.task(taskid)
			task.addEvent(ev)

			// retrieve parent task information
			if typ == trace.EvUserTaskCreate {
				if parentID := ev.Args[1]; parentID != 0 {
					parentTask := tasks.task(parentID)
					task.parent = parentTask
					if parentTask != nil {
						parentTask.children = append(parentTask.children, task)
					}
				}
			}

		case trace.EvGCStart:
			gcEvents = append(gcEvents, ev)
		}
	}
	// combine region info.
	analyzeGoroutines(res)
	for goid, stats := range gs {
		// gs is a global var defined in goroutines.go as a result
		// of analyzeGoroutines. TODO(hyangah): fix this not to depend
		// on a 'global' var.
		for _, s := range stats.Regions {
			if s.TaskID != 0 {
				task := tasks.task(s.TaskID)
				task.goroutines[goid] = struct{}{}
				task.regions = append(task.regions, regionDesc{UserRegionDesc: s, G: goid})
			}
			var frame trace.Frame
			if s.Start != nil {
				frame = *res.Stacks[s.Start.StkID][0]
			}
			id := regionTypeID{Frame: frame, Type: s.Name}
			regions[id] = append(regions[id], regionDesc{UserRegionDesc: s, G: goid})
		}
	}

	// sort regions in tasks based on the timestamps.
	for _, task := range tasks {
		sort.SliceStable(task.regions, func(i, j int) bool {
			si, sj := task.regions[i].firstTimestamp(), task.regions[j].firstTimestamp()
			if si != sj {
				return si < sj
			}
			return task.regions[i].lastTimestamp() < task.regions[j].lastTimestamp()
		})
	}
	return annotationAnalysisResult{tasks: tasks, regions: regions, gcEvents: gcEvents}, nil
}

// taskDesc represents a task.
type taskDesc struct {
	name       string              // user-provided task name
	id         uint64              // internal task id
	events     []*trace.Event      // sorted based on timestamp.
	regions    []regionDesc        // associated regions, sorted based on the start timestamp and then the last timestamp.
	goroutines map[uint64]struct{} // involved goroutines

	create *trace.Event // Task create event
	end    *trace.Event // Task end event

	parent   *taskDesc
	children []*taskDesc
}

func newTaskDesc(id uint64) *taskDesc {
	return &taskDesc{
		id:         id,
		goroutines: make(map[uint64]struct{}),
	}
}

func (task *taskDesc) String() string {
	if task == nil {
		return "task <nil>"
	}
	wb := new(bytes.Buffer)
	fmt.Fprintf(wb, "task %d:\t%s\n", task.id, task.name)
	fmt.Fprintf(wb, "\tstart: %v end: %v complete: %t\n", task.firstTimestamp(), task.endTimestamp(), task.complete())
	fmt.Fprintf(wb, "\t%d goroutines\n", len(task.goroutines))
	fmt.Fprintf(wb, "\t%d regions:\n", len(task.regions))
	for _, s := range task.regions {
		fmt.Fprintf(wb, "\t\t%s(goid=%d)\n", s.Name, s.G)
	}
	if task.parent != nil {
		fmt.Fprintf(wb, "\tparent: %s\n", task.parent.name)
	}
	fmt.Fprintf(wb, "\t%d children:\n", len(task.children))
	for _, c := range task.children {
		fmt.Fprintf(wb, "\t\t%s\n", c.name)
	}

	return wb.String()
}

// regionDesc represents a region.
type regionDesc struct {
	*trace.UserRegionDesc
	G uint64 // id of goroutine where the region was defined
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
		goroutines: make(map[uint64]struct{}),
	}
	tasks[taskID] = t
	return t
}

func (task *taskDesc) addEvent(ev *trace.Event) {
	if task == nil {
		return
	}

	task.events = append(task.events, ev)
	task.goroutines[ev.G] = struct{}{}

	switch typ := ev.Type; typ {
	case trace.EvUserTaskCreate:
		task.name = ev.SArgs[0]
		task.create = ev
	case trace.EvUserTaskEnd:
		task.end = ev
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

// descendants returns all the task nodes in the subtree rooted from this task.
func (task *taskDesc) descendants() []*taskDesc {
	if task == nil {
		return nil
	}
	res := []*taskDesc{task}
	for i := 0; len(res[i:]) > 0; i++ {
		t := res[i]
		for _, c := range t.children {
			res = append(res, c)
		}
	}
	return res
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
	endTs := task.endTimestamp()
	if last := task.lastEvent(); last != nil && last.Ts > endTs {
		return last.Ts
	}
	return endTs
}

// endTimestamp returns the timestamp of this task's end event.
// If the trace does not contain the task end event, the last
// timestamp of the trace will be returned.
func (task *taskDesc) endTimestamp() int64 {
	if task != nil && task.end != nil {
		return task.end.Ts
	}
	return lastTimestamp()
}

func (task *taskDesc) duration() time.Duration {
	return time.Duration(task.endTimestamp()-task.firstTimestamp()) * time.Nanosecond
}

func (region *regionDesc) duration() time.Duration {
	return time.Duration(region.lastTimestamp()-region.firstTimestamp()) * time.Nanosecond
}

// overlappingGCDuration returns the sum of GC period overlapping with the task's lifetime.
func (task *taskDesc) overlappingGCDuration(evs []*trace.Event) (overlapping time.Duration) {
	for _, ev := range evs {
		// make sure we only consider the global GC events.
		if typ := ev.Type; typ != trace.EvGCStart && typ != trace.EvGCSTWStart {
			continue
		}

		if o, overlapped := task.overlappingDuration(ev); overlapped {
			overlapping += o
		}
	}
	return overlapping
}

// overlappingInstant reports whether the instantaneous event, ev, occurred during
// any of the task's region if ev is a goroutine-local event, or overlaps with the
// task's lifetime if ev is a global event.
func (task *taskDesc) overlappingInstant(ev *trace.Event) bool {
	if _, ok := isUserAnnotationEvent(ev); ok && task.id != ev.Args[0] {
		return false // not this task's user event.
	}

	ts := ev.Ts
	taskStart := task.firstTimestamp()
	taskEnd := task.endTimestamp()
	if ts < taskStart || taskEnd < ts {
		return false
	}
	if ev.P == trace.GCP {
		return true
	}

	// Goroutine local event. Check whether there are regions overlapping with the event.
	goid := ev.G
	for _, region := range task.regions {
		if region.G != goid {
			continue
		}
		if region.firstTimestamp() <= ts && ts <= region.lastTimestamp() {
			return true
		}
	}
	return false
}

// overlappingDuration returns whether the durational event, ev, overlaps with
// any of the task's region if ev is a goroutine-local event, or overlaps with
// the task's lifetime if ev is a global event. It returns the overlapping time
// as well.
func (task *taskDesc) overlappingDuration(ev *trace.Event) (time.Duration, bool) {
	start := ev.Ts
	end := lastTimestamp()
	if ev.Link != nil {
		end = ev.Link.Ts
	}

	if start > end {
		return 0, false
	}

	goid := ev.G
	goid2 := ev.G
	if ev.Link != nil {
		goid2 = ev.Link.G
	}

	// This event is a global GC event
	if ev.P == trace.GCP {
		taskStart := task.firstTimestamp()
		taskEnd := task.endTimestamp()
		o := overlappingDuration(taskStart, taskEnd, start, end)
		return o, o > 0
	}

	// Goroutine local event. Check whether there are regions overlapping with the event.
	var overlapping time.Duration
	var lastRegionEnd int64 // the end of previous overlapping region
	for _, region := range task.regions {
		if region.G != goid && region.G != goid2 {
			continue
		}
		regionStart, regionEnd := region.firstTimestamp(), region.lastTimestamp()
		if regionStart < lastRegionEnd { // skip nested regions
			continue
		}

		if o := overlappingDuration(regionStart, regionEnd, start, end); o > 0 {
			// overlapping.
			lastRegionEnd = regionEnd
			overlapping += o
		}
	}
	return overlapping, overlapping > 0
}

// overlappingDuration returns the overlapping time duration between
// two time intervals [start1, end1] and [start2, end2] where
// start, end parameters are all int64 representing nanoseconds.
func overlappingDuration(start1, end1, start2, end2 int64) time.Duration {
	// assume start1 <= end1 and start2 <= end2
	if end1 < start2 || end2 < start1 {
		return 0
	}

	if start1 < start2 { // choose the later one
		start1 = start2
	}
	if end1 > end2 { // choose the earlier one
		end1 = end2
	}
	return time.Duration(end1 - start1)
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

// firstTimestamp returns the timestamp of region start event.
// If the region's start event is not present in the trace,
// the first timestamp of the trace will be returned.
func (region *regionDesc) firstTimestamp() int64 {
	if region.Start != nil {
		return region.Start.Ts
	}
	return firstTimestamp()
}

// lastTimestamp returns the timestamp of region end event.
// If the region's end event is not present in the trace,
// the last timestamp of the trace will be returned.
func (region *regionDesc) lastTimestamp() int64 {
	if region.End != nil {
		return region.End.Ts
	}
	return lastTimestamp()
}

// RelatedGoroutines returns IDs of goroutines related to the task. A goroutine
// is related to the task if user annotation activities for the task occurred.
// If non-zero depth is provided, this searches all events with BFS and includes
// goroutines unblocked any of related goroutines to the result.
func (task *taskDesc) RelatedGoroutines(events []*trace.Event, depth int) map[uint64]bool {
	start, end := task.firstTimestamp(), task.endTimestamp()

	gmap := map[uint64]bool{}
	for k := range task.goroutines {
		gmap[k] = true
	}

	for i := 0; i < depth; i++ {
		gmap1 := make(map[uint64]bool)
		for g := range gmap {
			gmap1[g] = true
		}
		for _, ev := range events {
			if ev.Ts < start || ev.Ts > end {
				continue
			}
			if ev.Type == trace.EvGoUnblock && gmap[ev.Args[0]] {
				gmap1[ev.G] = true
			}
			gmap = gmap1
		}
	}
	gmap[0] = true // for GC events (goroutine id = 0)
	return gmap
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
	if text := r.FormValue("logtext"); text != "" {
		name = append(name, fmt.Sprintf("log contains %q", text))
		conditions = append(conditions, func(t *taskDesc) bool {
			return taskMatches(t, text)
		})
	}

	return &taskFilter{name: strings.Join(name, ","), cond: conditions}, nil
}

func taskMatches(t *taskDesc, text string) bool {
	for _, ev := range t.events {
		switch ev.Type {
		case trace.EvUserTaskCreate, trace.EvUserRegion, trace.EvUserLog:
			for _, s := range ev.SArgs {
				if strings.Contains(s, text) {
					return true
				}
			}
		}
	}
	return false
}

type regionFilter struct {
	name string
	cond []func(regionTypeID, regionDesc) bool
}

func (f *regionFilter) match(id regionTypeID, s regionDesc) bool {
	for _, c := range f.cond {
		if !c(id, s) {
			return false
		}
	}
	return true
}

func newRegionFilter(r *http.Request) (*regionFilter, error) {
	if err := r.ParseForm(); err != nil {
		return nil, err
	}

	var name []string
	var conditions []func(regionTypeID, regionDesc) bool

	param := r.Form
	if typ, ok := param["type"]; ok && len(typ) > 0 {
		name = append(name, "type="+typ[0])
		conditions = append(conditions, func(id regionTypeID, s regionDesc) bool {
			return id.Type == typ[0]
		})
	}
	if pc, err := strconv.ParseUint(r.FormValue("pc"), 16, 64); err == nil {
		name = append(name, fmt.Sprintf("pc=%x", pc))
		conditions = append(conditions, func(id regionTypeID, s regionDesc) bool {
			return id.Frame.PC == pc
		})
	}

	if lat, err := time.ParseDuration(r.FormValue("latmin")); err == nil {
		name = append(name, fmt.Sprintf("latency >= %s", lat))
		conditions = append(conditions, func(_ regionTypeID, s regionDesc) bool {
			return s.duration() >= lat
		})
	}
	if lat, err := time.ParseDuration(r.FormValue("latmax")); err == nil {
		name = append(name, fmt.Sprintf("latency <= %s", lat))
		conditions = append(conditions, func(_ regionTypeID, s regionDesc) bool {
			return s.duration() <= lat
		})
	}

	return &regionFilter{name: strings.Join(name, ","), cond: conditions}, nil
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
		if h.Buckets[i] > 0 {
			fmt.Fprintf(w, `<tr><td class="histoTime" align="right"><a href=%s>%s</a></td>`, urlmaker(h.BucketMin(i), h.BucketMin(i+1)), niceDuration(h.BucketMin(i)))
		} else {
			fmt.Fprintf(w, `<tr><td class="histoTime" align="right">%s</td>`, niceDuration(h.BucketMin(i)))
		}
		// Bucket bar.
		width := h.Buckets[i] * barWidth / maxCount
		fmt.Fprintf(w, `<td><div style="width:%dpx;background:blue;position:relative">&nbsp;</div></td>`, width)
		// Bucket count.
		fmt.Fprintf(w, `<td align="right"><div style="position:relative">%d</div></td>`, h.Buckets[i])
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

type regionStats struct {
	regionTypeID
	Histogram durationHistogram
}

func (s *regionStats) UserRegionURL() func(min, max time.Duration) string {
	return func(min, max time.Duration) string {
		return fmt.Sprintf("/userregion?type=%s&pc=%x&latmin=%v&latmax=%v", template.URLQueryEscaper(s.Type), s.Frame.PC, template.URLQueryEscaper(min), template.URLQueryEscaper(max))
	}
}

func (s *regionStats) add(region regionDesc) {
	s.Histogram.add(region.duration())
}

var templUserRegionTypes = template.Must(template.New("").Parse(`
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
<th>Region type</th>
<th>Count</th>
<th>Duration distribution (complete tasks)</th>
</tr>
{{range $}}
  <tr>
    <td>{{.Type}}<br>{{.Frame.Fn}}<br>{{.Frame.File}}:{{.Frame.Line}}</td>
    <td><a href="/userregion?type={{.Type}}&pc={{.Frame.PC}}">{{.Histogram.Count}}</a></td>
    <td>{{.Histogram.ToHTML (.UserRegionURL)}}</td>
  </tr>
{{end}}
</table>
</body>
</html>
`))

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

Search log text: <form onsubmit="window.location.search+='&logtext='+window.logtextinput.value; return false">
<input name="logtext" id="logtextinput" type="text"><input type="submit">
</form><br>

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
	<tr>
		<td></td>
		<td></td>
		<td></td>
		<td>GC:{{$el.GCTime}}</td>
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

func formatUserLog(ev *trace.Event) string {
	k, v := ev.SArgs[0], ev.SArgs[1]
	if k == "" {
		return v
	}
	if v == "" {
		return k
	}
	return fmt.Sprintf("%v=%v", k, v)
}

func describeEvent(ev *trace.Event) string {
	switch ev.Type {
	case trace.EvGoCreate:
		goid := ev.Args[0]
		return fmt.Sprintf("new goroutine %d: %s", goid, gs[goid].Name)
	case trace.EvGoEnd, trace.EvGoStop:
		return "goroutine stopped"
	case trace.EvUserLog:
		return formatUserLog(ev)
	case trace.EvUserRegion:
		if ev.Args[1] == 0 {
			duration := "unknown"
			if ev.Link != nil {
				duration = (time.Duration(ev.Link.Ts-ev.Ts) * time.Nanosecond).String()
			}
			return fmt.Sprintf("region %s started (duration: %v)", ev.SArgs[0], duration)
		}
		return fmt.Sprintf("region %s ended", ev.SArgs[0])
	case trace.EvUserTaskCreate:
		return fmt.Sprintf("task %v (id %d, parent %d) created", ev.SArgs[0], ev.Args[0], ev.Args[1])
		// TODO: add child task creation events into the parent task events
	case trace.EvUserTaskEnd:
		return "task end"
	}
	return ""
}

func isUserAnnotationEvent(ev *trace.Event) (taskID uint64, ok bool) {
	switch ev.Type {
	case trace.EvUserLog, trace.EvUserRegion, trace.EvUserTaskCreate, trace.EvUserTaskEnd:
		return ev.Args[0], true
	}
	return 0, false
}

var templUserRegionType = template.Must(template.New("").Funcs(template.FuncMap{
	"prettyDuration": func(nsec int64) template.HTML {
		d := time.Duration(nsec) * time.Nanosecond
		return template.HTML(niceDuration(d))
	},
	"percent": func(dividened, divisor int64) template.HTML {
		if divisor == 0 {
			return ""
		}
		return template.HTML(fmt.Sprintf("(%.1f%%)", float64(dividened)/float64(divisor)*100))
	},
	"barLen": func(dividened, divisor int64) template.HTML {
		if divisor == 0 {
			return "0"
		}
		return template.HTML(fmt.Sprintf("%.2f%%", float64(dividened)/float64(divisor)*100))
	},
	"unknownTime": func(desc regionDesc) int64 {
		sum := desc.ExecTime + desc.IOTime + desc.BlockTime + desc.SyscallTime + desc.SchedWaitTime
		if sum < desc.TotalTime {
			return desc.TotalTime - sum
		}
		return 0
	},
}).Parse(`
<!DOCTYPE html>
<title>Goroutine {{.Name}}</title>
<style>
th {
  background-color: #050505;
  color: #fff;
}
table {
  border-collapse: collapse;
}
.details tr:hover {
  background-color: #f2f2f2;
}
.details td {
  text-align: right;
  border: 1px solid #000;
}
.details td.id {
  text-align: left;
}
.stacked-bar-graph {
  width: 300px;
  height: 10px;
  color: #414042;
  white-space: nowrap;
  font-size: 5px;
}
.stacked-bar-graph span {
  display: inline-block;
  width: 100%;
  height: 100%;
  box-sizing: border-box;
  float: left;
  padding: 0;
}
.unknown-time { background-color: #636363; }
.exec-time { background-color: #d7191c; }
.io-time { background-color: #fdae61; }
.block-time { background-color: #d01c8b; }
.syscall-time { background-color: #7b3294; }
.sched-time { background-color: #2c7bb6; }
</style>

<script>
function reloadTable(key, value) {
  let params = new URLSearchParams(window.location.search);
  params.set(key, value);
  window.location.search = params.toString();
}
</script>

<h2>{{.Name}}</h2>

<table class="details">
<tr>
<th> Goroutine </th>
<th> Task </th>
<th onclick="reloadTable('sortby', 'TotalTime')"> Total</th>
<th></th>
<th onclick="reloadTable('sortby', 'ExecTime')" class="exec-time"> Execution</th>
<th onclick="reloadTable('sortby', 'IOTime')" class="io-time"> Network wait</th>
<th onclick="reloadTable('sortby', 'BlockTime')" class="block-time"> Sync block </th>
<th onclick="reloadTable('sortby', 'SyscallTime')" class="syscall-time"> Blocking syscall</th>
<th onclick="reloadTable('sortby', 'SchedWaitTime')" class="sched-time"> Scheduler wait</th>
<th onclick="reloadTable('sortby', 'SweepTime')"> GC sweeping</th>
<th onclick="reloadTable('sortby', 'GCTime')"> GC pause</th>
</tr>
{{range .Data}}
  <tr>
    <td> <a href="/trace?goid={{.G}}">{{.G}}</a> </td>
    <td> {{if .TaskID}}<a href="/trace?taskid={{.TaskID}}">{{.TaskID}}</a>{{end}} </td>
    <td> {{prettyDuration .TotalTime}} </td>
    <td>
        <div class="stacked-bar-graph">
          {{if unknownTime .}}<span style="width:{{barLen (unknownTime .) $.MaxTotal}}" class="unknown-time">&nbsp;</span>{{end}}
          {{if .ExecTime}}<span style="width:{{barLen .ExecTime $.MaxTotal}}" class="exec-time">&nbsp;</span>{{end}}
          {{if .IOTime}}<span style="width:{{barLen .IOTime $.MaxTotal}}" class="io-time">&nbsp;</span>{{end}}
          {{if .BlockTime}}<span style="width:{{barLen .BlockTime $.MaxTotal}}" class="block-time">&nbsp;</span>{{end}}
          {{if .SyscallTime}}<span style="width:{{barLen .SyscallTime $.MaxTotal}}" class="syscall-time">&nbsp;</span>{{end}}
          {{if .SchedWaitTime}}<span style="width:{{barLen .SchedWaitTime $.MaxTotal}}" class="sched-time">&nbsp;</span>{{end}}
        </div>
    </td>
    <td> {{prettyDuration .ExecTime}}</td>
    <td> {{prettyDuration .IOTime}}</td>
    <td> {{prettyDuration .BlockTime}}</td>
    <td> {{prettyDuration .SyscallTime}}</td>
    <td> {{prettyDuration .SchedWaitTime}}</td>
    <td> {{prettyDuration .SweepTime}} {{percent .SweepTime .TotalTime}}</td>
    <td> {{prettyDuration .GCTime}} {{percent .GCTime .TotalTime}}</td>
  </tr>
{{end}}
</table>
`))

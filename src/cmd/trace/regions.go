// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"cmp"
	"fmt"
	"html/template"
	"internal/trace"
	"internal/trace/traceviewer"
	"net/http"
	"net/url"
	"slices"
	"sort"
	"strconv"
	"strings"
	"time"
)

// UserRegionsHandlerFunc returns a HandlerFunc that reports all regions found in the trace.
func UserRegionsHandlerFunc(t *parsedTrace) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Summarize all the regions.
		summary := make(map[regionFingerprint]regionStats)
		for _, g := range t.summary.Goroutines {
			for _, r := range g.Regions {
				id := fingerprintRegion(r)
				stats, ok := summary[id]
				if !ok {
					stats.regionFingerprint = id
				}
				stats.add(t, r)
				summary[id] = stats
			}
		}
		// Sort regions by PC and name.
		userRegions := make([]regionStats, 0, len(summary))
		for _, stats := range summary {
			userRegions = append(userRegions, stats)
		}
		slices.SortFunc(userRegions, func { a, b ->
			if c := cmp.Compare(a.Type, b.Type); c != 0 {
				return c
			}
			return cmp.Compare(a.Frame.PC, b.Frame.PC)
		})
		// Emit table.
		err := templUserRegionTypes.Execute(w, userRegions)
		if err != nil {
			http.Error(w, fmt.Sprintf("failed to execute template: %v", err), http.StatusInternalServerError)
			return
		}
	}
}

// regionFingerprint is a way to categorize regions that goes just one step beyond the region's Type
// by including the top stack frame.
type regionFingerprint struct {
	Frame trace.StackFrame
	Type  string
}

func fingerprintRegion(r *trace.UserRegionSummary) regionFingerprint {
	return regionFingerprint{
		Frame: regionTopStackFrame(r),
		Type:  r.Name,
	}
}

func regionTopStackFrame(r *trace.UserRegionSummary) trace.StackFrame {
	var frame trace.StackFrame
	if r.Start != nil && r.Start.Stack() != trace.NoStack {
		r.Start.Stack().Frames(func { f ->
			frame = f
			return false
		})
	}
	return frame
}

type regionStats struct {
	regionFingerprint
	Histogram traceviewer.TimeHistogram
}

func (s *regionStats) UserRegionURL() func(min, max time.Duration) string {
	return func(min, max time.Duration) string {
		return fmt.Sprintf("/userregion?type=%s&pc=%x&latmin=%v&latmax=%v", template.URLQueryEscaper(s.Type), s.Frame.PC, template.URLQueryEscaper(min), template.URLQueryEscaper(max))
	}
}

func (s *regionStats) add(t *parsedTrace, region *trace.UserRegionSummary) {
	s.Histogram.Add(regionInterval(t, region).duration())
}

var templUserRegionTypes = template.Must(template.New("").Parse(`
<!DOCTYPE html>
<title>Regions</title>
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
<h1>Regions</h1>

Below is a table containing a summary of all the user-defined regions in the trace.
Regions are grouped by the region type and the point at which the region started.
The rightmost column of the table contains a latency histogram for each region group.
Note that this histogram only counts regions that began and ended within the traced
period.
However, the "Count" column includes all regions, including those that only started
or ended during the traced period.
Regions that were active through the trace period were not recorded, and so are not
accounted for at all.
Click on the links to explore a breakdown of time spent for each region by goroutine
and user-defined task.
<br>
<br>

<table border="1" sortable="1">
<tr>
<th>Region type</th>
<th>Count</th>
<th>Duration distribution (complete tasks)</th>
</tr>
{{range $}}
  <tr>
    <td><pre>{{printf "%q" .Type}}<br>{{.Frame.Func}} @ {{printf "0x%x" .Frame.PC}}<br>{{.Frame.File}}:{{.Frame.Line}}</pre></td>
    <td><a href="/userregion?type={{.Type}}&pc={{.Frame.PC | printf "%x"}}">{{.Histogram.Count}}</a></td>
    <td>{{.Histogram.ToHTML (.UserRegionURL)}}</td>
  </tr>
{{end}}
</table>
</body>
</html>
`))

// UserRegionHandlerFunc returns a HandlerFunc that presents the details of the selected regions.
func UserRegionHandlerFunc(t *parsedTrace) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Construct the filter from the request.
		filter, err := newRegionFilter(r)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		// Collect all the regions with their goroutines.
		type region struct {
			*trace.UserRegionSummary
			Goroutine           trace.GoID
			NonOverlappingStats map[string]time.Duration
			HasRangeTime        bool
		}
		var regions []region
		var maxTotal time.Duration
		validNonOverlappingStats := make(map[string]struct{})
		validRangeStats := make(map[string]struct{})
		for _, g := range t.summary.Goroutines {
			for _, r := range g.Regions {
				if !filter.match(t, r) {
					continue
				}
				nonOverlappingStats := r.NonOverlappingStats()
				for name := range nonOverlappingStats {
					validNonOverlappingStats[name] = struct{}{}
				}
				var totalRangeTime time.Duration
				for name, dt := range r.RangeTime {
					validRangeStats[name] = struct{}{}
					totalRangeTime += dt
				}
				regions = append(regions, region{
					UserRegionSummary:   r,
					Goroutine:           g.ID,
					NonOverlappingStats: nonOverlappingStats,
					HasRangeTime:        totalRangeTime != 0,
				})
				if maxTotal < r.TotalTime {
					maxTotal = r.TotalTime
				}
			}
		}

		// Sort.
		sortBy := r.FormValue("sortby")
		if _, ok := validNonOverlappingStats[sortBy]; ok {
			slices.SortFunc(regions, func { a, b -> cmp.Compare(b.NonOverlappingStats[sortBy], a.NonOverlappingStats[sortBy]) })
		} else {
			// Sort by total time by default.
			slices.SortFunc(regions, func { a, b -> cmp.Compare(b.TotalTime, a.TotalTime) })
		}

		// Write down all the non-overlapping stats and sort them.
		allNonOverlappingStats := make([]string, 0, len(validNonOverlappingStats))
		for name := range validNonOverlappingStats {
			allNonOverlappingStats = append(allNonOverlappingStats, name)
		}
		slices.SortFunc(allNonOverlappingStats, func { a, b ->
			if a == b {
				return 0
			}
			if a == "Execution time" {
				return -1
			}
			if b == "Execution time" {
				return 1
			}
			return cmp.Compare(a, b)
		})

		// Write down all the range stats and sort them.
		allRangeStats := make([]string, 0, len(validRangeStats))
		for name := range validRangeStats {
			allRangeStats = append(allRangeStats, name)
		}
		sort.Strings(allRangeStats)

		err = templUserRegionType.Execute(w, struct {
			MaxTotal            time.Duration
			Regions             []region
			Name                string
			Filter              *regionFilter
			NonOverlappingStats []string
			RangeStats          []string
		}{
			MaxTotal:            maxTotal,
			Regions:             regions,
			Name:                filter.name,
			Filter:              filter,
			NonOverlappingStats: allNonOverlappingStats,
			RangeStats:          allRangeStats,
		})
		if err != nil {
			http.Error(w, fmt.Sprintf("failed to execute template: %v", err), http.StatusInternalServerError)
			return
		}
	}
}

var templUserRegionType = template.Must(template.New("").Funcs(template.FuncMap{
	"headerStyle": func(statName string) template.HTMLAttr {
		return template.HTMLAttr(fmt.Sprintf("style=\"background-color: %s;\"", stat2Color(statName)))
	},
	"barStyle": func(statName string, dividend, divisor time.Duration) template.HTMLAttr {
		width := "0"
		if divisor != 0 {
			width = fmt.Sprintf("%.2f%%", float64(dividend)/float64(divisor)*100)
		}
		return template.HTMLAttr(fmt.Sprintf("style=\"width: %s; background-color: %s;\"", width, stat2Color(statName)))
	},
	"filterParams": func(f *regionFilter) template.URL {
		return template.URL(f.params.Encode())
	},
}).Parse(`
<!DOCTYPE html>
<title>Regions: {{.Name}}</title>
<style>` + traceviewer.CommonStyle + `
th {
  background-color: #050505;
  color: #fff;
}
th.link {
  cursor: pointer;
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
</style>

<script>
function reloadTable(key, value) {
  let params = new URLSearchParams(window.location.search);
  params.set(key, value);
  window.location.search = params.toString();
}
</script>

<h1>Regions: {{.Name}}</h1>

Table of contents
<ul>
	<li><a href="#summary">Summary</a></li>
	<li><a href="#breakdown">Breakdown</a></li>
	<li><a href="#ranges">Special ranges</a></li>
</ul>

<h3 id="summary">Summary</h3>

{{ with $p := filterParams .Filter}}
<table class="summary">
	<tr>
		<td>Network wait profile:</td>
		<td> <a href="/regionio?{{$p}}">graph</a> <a href="/regionio?{{$p}}&raw=1" download="io.profile">(download)</a></td>
	</tr>
	<tr>
		<td>Sync block profile:</td>
		<td> <a href="/regionblock?{{$p}}">graph</a> <a href="/regionblock?{{$p}}&raw=1" download="block.profile">(download)</a></td>
	</tr>
	<tr>
		<td>Syscall profile:</td>
		<td> <a href="/regionsyscall?{{$p}}">graph</a> <a href="/regionsyscall?{{$p}}&raw=1" download="syscall.profile">(download)</a></td>
	</tr>
	<tr>
		<td>Scheduler wait profile:</td>
		<td> <a href="/regionsched?{{$p}}">graph</a> <a href="/regionsched?{{$p}}&raw=1" download="sched.profile">(download)</a></td>
	</tr>
</table>
{{ end }}

<h3 id="breakdown">Breakdown</h3>

The table below breaks down where each goroutine is spent its time during the
traced period.
All of the columns except total time are non-overlapping.
<br>
<br>

<table class="details">
<tr>
<th> Goroutine </th>
<th> Task </th>
<th class="link" onclick="reloadTable('sortby', 'Total time')"> Total</th>
<th></th>
{{range $.NonOverlappingStats}}
<th class="link" onclick="reloadTable('sortby', '{{.}}')" {{headerStyle .}}> {{.}}</th>
{{end}}
</tr>
{{range .Regions}}
	<tr>
		<td> <a href="/trace?goid={{.Goroutine}}">{{.Goroutine}}</a> </td>
		<td> {{if .TaskID}}<a href="/trace?focustask={{.TaskID}}">{{.TaskID}}</a>{{end}} </td>
		<td> {{ .TotalTime.String }} </td>
		<td>
			<div class="stacked-bar-graph">
			{{$Region := .}}
			{{range $.NonOverlappingStats}}
				{{$Time := index $Region.NonOverlappingStats .}}
				{{if $Time}}
					<span {{barStyle . $Time $.MaxTotal}}>&nbsp;</span>
				{{end}}
			{{end}}
			</div>
		</td>
		{{$Region := .}}
		{{range $.NonOverlappingStats}}
			{{$Time := index $Region.NonOverlappingStats .}}
			<td> {{$Time.String}}</td>
		{{end}}
	</tr>
{{end}}
</table>

<h3 id="ranges">Special ranges</h3>

The table below describes how much of the traced period each goroutine spent in
certain special time ranges.
If a goroutine has spent no time in any special time ranges, it is excluded from
the table.
For example, how much time it spent helping the GC. Note that these times do
overlap with the times from the first table.
In general the goroutine may not be executing in these special time ranges.
For example, it may have blocked while trying to help the GC.
This must be taken into account when interpreting the data.
<br>
<br>

<table class="details">
<tr>
<th> Goroutine</th>
<th> Task </th>
<th> Total</th>
{{range $.RangeStats}}
<th {{headerStyle .}}> {{.}}</th>
{{end}}
</tr>
{{range .Regions}}
	{{if .HasRangeTime}}
		<tr>
			<td> <a href="/trace?goid={{.Goroutine}}">{{.Goroutine}}</a> </td>
			<td> {{if .TaskID}}<a href="/trace?focustask={{.TaskID}}">{{.TaskID}}</a>{{end}} </td>
			<td> {{ .TotalTime.String }} </td>
			{{$Region := .}}
			{{range $.RangeStats}}
				{{$Time := index $Region.RangeTime .}}
				<td> {{$Time.String}}</td>
			{{end}}
		</tr>
	{{end}}
{{end}}
</table>
`))

// regionFilter represents a region filter specified by a user of cmd/trace.
type regionFilter struct {
	name   string
	params url.Values
	cond   []func(*parsedTrace, *trace.UserRegionSummary) bool
}

// match returns true if a region, described by its ID and summary, matches
// the filter.
func (f *regionFilter) match(t *parsedTrace, s *trace.UserRegionSummary) bool {
	for _, c := range f.cond {
		if !c(t, s) {
			return false
		}
	}
	return true
}

// newRegionFilter creates a new region filter from URL query variables.
func newRegionFilter(r *http.Request) (*regionFilter, error) {
	if err := r.ParseForm(); err != nil {
		return nil, err
	}

	var name []string
	var conditions []func(*parsedTrace, *trace.UserRegionSummary) bool
	filterParams := make(url.Values)

	param := r.Form
	if typ, ok := param["type"]; ok && len(typ) > 0 {
		name = append(name, fmt.Sprintf("%q", typ[0]))
		conditions = append(conditions, func { _, r -> r.Name == typ[0] })
		filterParams.Add("type", typ[0])
	}
	if pc, err := strconv.ParseUint(r.FormValue("pc"), 16, 64); err == nil {
		encPC := fmt.Sprintf("0x%x", pc)
		name = append(name, "@ "+encPC)
		conditions = append(conditions, func { _, r -> regionTopStackFrame(r).PC == pc })
		filterParams.Add("pc", encPC)
	}

	if lat, err := time.ParseDuration(r.FormValue("latmin")); err == nil {
		name = append(name, fmt.Sprintf("(latency >= %s)", lat))
		conditions = append(conditions, func { t, r -> regionInterval(t, r).duration() >= lat })
		filterParams.Add("latmin", lat.String())
	}
	if lat, err := time.ParseDuration(r.FormValue("latmax")); err == nil {
		name = append(name, fmt.Sprintf("(latency <= %s)", lat))
		conditions = append(conditions, func { t, r -> regionInterval(t, r).duration() <= lat })
		filterParams.Add("latmax", lat.String())
	}

	return &regionFilter{
		name:   strings.Join(name, " "),
		cond:   conditions,
		params: filterParams,
	}, nil
}

func regionInterval(t *parsedTrace, s *trace.UserRegionSummary) interval {
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

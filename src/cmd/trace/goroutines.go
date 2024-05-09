// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Goroutine-related profiles.

package main

import (
	"cmp"
	"fmt"
	"html/template"
	"internal/trace"
	"internal/trace/traceviewer"
	"log"
	"net/http"
	"slices"
	"sort"
	"strings"
	"time"
)

// GoroutinesHandlerFunc returns a HandlerFunc that serves list of goroutine groups.
func GoroutinesHandlerFunc(summaries map[trace.GoID]*trace.GoroutineSummary) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// goroutineGroup describes a group of goroutines grouped by name.
		type goroutineGroup struct {
			Name     string        // Start function.
			N        int           // Total number of goroutines in this group.
			ExecTime time.Duration // Total execution time of all goroutines in this group.
		}
		// Accumulate groups by Name.
		groupsByName := make(map[string]goroutineGroup)
		for _, summary := range summaries {
			group := groupsByName[summary.Name]
			group.Name = summary.Name
			group.N++
			group.ExecTime += summary.ExecTime
			groupsByName[summary.Name] = group
		}
		var groups []goroutineGroup
		for _, group := range groupsByName {
			groups = append(groups, group)
		}
		slices.SortFunc(groups, func(a, b goroutineGroup) int {
			return cmp.Compare(b.ExecTime, a.ExecTime)
		})
		w.Header().Set("Content-Type", "text/html;charset=utf-8")
		if err := templGoroutines.Execute(w, groups); err != nil {
			log.Printf("failed to execute template: %v", err)
			return
		}
	}
}

var templGoroutines = template.Must(template.New("").Parse(`
<html>
<style>` + traceviewer.CommonStyle + `
table {
  border-collapse: collapse;
}
td,
th {
  border: 1px solid black;
  padding-left: 8px;
  padding-right: 8px;
  padding-top: 4px;
  padding-bottom: 4px;
}
</style>
<body>
<h1>Goroutines</h1>
Below is a table of all goroutines in the trace grouped by start location and sorted by the total execution time of the group.<br>
<br>
Click a start location to view more details about that group.<br>
<br>
<table>
  <tr>
    <th>Start location</th>
	<th>Count</th>
	<th>Total execution time</th>
  </tr>
{{range $}}
  <tr>
    <td><code><a href="/goroutine?name={{.Name}}">{{or .Name "(Inactive, no stack trace sampled)"}}</a></code></td>
	<td>{{.N}}</td>
	<td>{{.ExecTime}}</td>
  </tr>
{{end}}
</table>
</body>
</html>
`))

// GoroutineHandler creates a handler that serves information about
// goroutines in a particular group.
func GoroutineHandler(summaries map[trace.GoID]*trace.GoroutineSummary) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		goroutineName := r.FormValue("name")

		type goroutine struct {
			*trace.GoroutineSummary
			NonOverlappingStats map[string]time.Duration
			HasRangeTime        bool
		}

		// Collect all the goroutines in the group.
		var (
			goroutines              []goroutine
			name                    string
			totalExecTime, execTime time.Duration
			maxTotalTime            time.Duration
		)
		validNonOverlappingStats := make(map[string]struct{})
		validRangeStats := make(map[string]struct{})
		for _, summary := range summaries {
			totalExecTime += summary.ExecTime

			if summary.Name != goroutineName {
				continue
			}
			nonOverlappingStats := summary.NonOverlappingStats()
			for name := range nonOverlappingStats {
				validNonOverlappingStats[name] = struct{}{}
			}
			var totalRangeTime time.Duration
			for name, dt := range summary.RangeTime {
				validRangeStats[name] = struct{}{}
				totalRangeTime += dt
			}
			goroutines = append(goroutines, goroutine{
				GoroutineSummary:    summary,
				NonOverlappingStats: nonOverlappingStats,
				HasRangeTime:        totalRangeTime != 0,
			})
			name = summary.Name
			execTime += summary.ExecTime
			if maxTotalTime < summary.TotalTime {
				maxTotalTime = summary.TotalTime
			}
		}

		// Compute the percent of total execution time these goroutines represent.
		execTimePercent := ""
		if totalExecTime > 0 {
			execTimePercent = fmt.Sprintf("%.2f%%", float64(execTime)/float64(totalExecTime)*100)
		}

		// Sort.
		sortBy := r.FormValue("sortby")
		if _, ok := validNonOverlappingStats[sortBy]; ok {
			slices.SortFunc(goroutines, func(a, b goroutine) int {
				return cmp.Compare(b.NonOverlappingStats[sortBy], a.NonOverlappingStats[sortBy])
			})
		} else {
			// Sort by total time by default.
			slices.SortFunc(goroutines, func(a, b goroutine) int {
				return cmp.Compare(b.TotalTime, a.TotalTime)
			})
		}

		// Write down all the non-overlapping stats and sort them.
		allNonOverlappingStats := make([]string, 0, len(validNonOverlappingStats))
		for name := range validNonOverlappingStats {
			allNonOverlappingStats = append(allNonOverlappingStats, name)
		}
		slices.SortFunc(allNonOverlappingStats, func(a, b string) int {
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

		err := templGoroutine.Execute(w, struct {
			Name                string
			N                   int
			ExecTimePercent     string
			MaxTotal            time.Duration
			Goroutines          []goroutine
			NonOverlappingStats []string
			RangeStats          []string
		}{
			Name:                name,
			N:                   len(goroutines),
			ExecTimePercent:     execTimePercent,
			MaxTotal:            maxTotalTime,
			Goroutines:          goroutines,
			NonOverlappingStats: allNonOverlappingStats,
			RangeStats:          allRangeStats,
		})
		if err != nil {
			http.Error(w, fmt.Sprintf("failed to execute template: %v", err), http.StatusInternalServerError)
			return
		}
	}
}

func stat2Color(statName string) string {
	color := "#636363"
	if strings.HasPrefix(statName, "Block time") {
		color = "#d01c8b"
	}
	switch statName {
	case "Sched wait time":
		color = "#2c7bb6"
	case "Syscall execution time":
		color = "#7b3294"
	case "Execution time":
		color = "#d7191c"
	}
	return color
}

var templGoroutine = template.Must(template.New("").Funcs(template.FuncMap{
	"percent": func(dividend, divisor time.Duration) template.HTML {
		if divisor == 0 {
			return ""
		}
		return template.HTML(fmt.Sprintf("(%.1f%%)", float64(dividend)/float64(divisor)*100))
	},
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
}).Parse(`
<!DOCTYPE html>
<title>Goroutines: {{.Name}}</title>
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
  border: 1px solid black;
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

<h1>Goroutines</h1>

Table of contents
<ul>
	<li><a href="#summary">Summary</a></li>
	<li><a href="#breakdown">Breakdown</a></li>
	<li><a href="#ranges">Special ranges</a></li>
</ul>

<h3 id="summary">Summary</h3>

<table class="summary">
	<tr>
		<td>Goroutine start location:</td>
		<td><code>{{.Name}}</code></td>
	</tr>
	<tr>
		<td>Count:</td>
		<td>{{.N}}</td>
	</tr>
	<tr>
		<td>Execution Time:</td>
		<td>{{.ExecTimePercent}} of total program execution time </td>
	</tr>
	<tr>
		<td>Network wait profile:</td>
		<td> <a href="/io?name={{.Name}}">graph</a> <a href="/io?name={{.Name}}&raw=1" download="io.profile">(download)</a></td>
	</tr>
	<tr>
		<td>Sync block profile:</td>
		<td> <a href="/block?name={{.Name}}">graph</a> <a href="/block?name={{.Name}}&raw=1" download="block.profile">(download)</a></td>
	</tr>
	<tr>
		<td>Syscall profile:</td>
		<td> <a href="/syscall?name={{.Name}}">graph</a> <a href="/syscall?name={{.Name}}&raw=1" download="syscall.profile">(download)</a></td>
		</tr>
	<tr>
		<td>Scheduler wait profile:</td>
		<td> <a href="/sched?name={{.Name}}">graph</a> <a href="/sched?name={{.Name}}&raw=1" download="sched.profile">(download)</a></td>
	</tr>
</table>

<h3 id="breakdown">Breakdown</h3>

The table below breaks down where each goroutine is spent its time during the
traced period.
All of the columns except total time are non-overlapping.
<br>
<br>

<table class="details">
<tr>
<th> Goroutine</th>
<th class="link" onclick="reloadTable('sortby', 'Total time')"> Total</th>
<th></th>
{{range $.NonOverlappingStats}}
<th class="link" onclick="reloadTable('sortby', '{{.}}')" {{headerStyle .}}> {{.}}</th>
{{end}}
</tr>
{{range .Goroutines}}
	<tr>
		<td> <a href="/trace?goid={{.ID}}">{{.ID}}</a> </td>
		<td> {{ .TotalTime.String }} </td>
		<td>
			<div class="stacked-bar-graph">
			{{$Goroutine := .}}
			{{range $.NonOverlappingStats}}
				{{$Time := index $Goroutine.NonOverlappingStats .}}
				{{if $Time}}
					<span {{barStyle . $Time $.MaxTotal}}>&nbsp;</span>
				{{end}}
			{{end}}
			</div>
		</td>
		{{$Goroutine := .}}
		{{range $.NonOverlappingStats}}
			{{$Time := index $Goroutine.NonOverlappingStats .}}
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
<th> Total</th>
{{range $.RangeStats}}
<th {{headerStyle .}}> {{.}}</th>
{{end}}
</tr>
{{range .Goroutines}}
	{{if .HasRangeTime}}
		<tr>
			<td> <a href="/trace?goid={{.ID}}">{{.ID}}</a> </td>
			<td> {{ .TotalTime.String }} </td>
			{{$Goroutine := .}}
			{{range $.RangeStats}}
				{{$Time := index $Goroutine.RangeTime .}}
				<td> {{$Time.String}}</td>
			{{end}}
		</tr>
	{{end}}
{{end}}
</table>
`))

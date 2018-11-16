// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Goroutine-related profiles.

package main

import (
	"fmt"
	"html/template"
	"log"
	"net/http"
	"reflect"
	"sort"
	"strconv"
	"sync"
	"time"

	trace "internal/traceparser"
)

func init() {
	http.HandleFunc("/goroutines", httpGoroutines)
	http.HandleFunc("/goroutine", httpGoroutine)
}

// gtype describes a group of goroutines grouped by start PC.
type gtype struct {
	ID       uint64 // Unique identifier (PC).
	Name     string // Start function.
	N        int    // Total number of goroutines in this group.
	ExecTime int64  // Total execution time of all goroutines in this group.
}

var (
	gsInit sync.Once
	gs     map[uint64]*trace.GDesc
)

// analyzeGoroutines generates statistics about execution of all goroutines and stores them in gs.
func analyzeGoroutines(res *trace.Parsed) {
	gsInit.Do(func() {
		gs = res.GoroutineStats()
	})
}

// httpGoroutines serves list of goroutine groups.
func httpGoroutines(w http.ResponseWriter, r *http.Request) {
	events, err := parseTrace()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	analyzeGoroutines(events)
	gss := make(map[uint64]gtype)
	for _, g := range gs {
		gs1 := gss[g.PC]
		gs1.ID = g.PC
		gs1.Name = g.Name
		gs1.N++
		gs1.ExecTime += g.ExecTime
		gss[g.PC] = gs1
	}
	var glist []gtype
	for k, v := range gss {
		v.ID = k
		glist = append(glist, v)
	}
	sort.Slice(glist, func(i, j int) bool { return glist[i].ExecTime > glist[j].ExecTime })
	w.Header().Set("Content-Type", "text/html;charset=utf-8")
	if err := templGoroutines.Execute(w, glist); err != nil {
		log.Printf("failed to execute template: %v", err)
		return
	}
}

var templGoroutines = template.Must(template.New("").Parse(`
<html>
<body>
Goroutines: <br>
{{range $}}
  <a href="/goroutine?id={{.ID}}">{{.Name}}</a> N={{.N}} <br>
{{end}}
</body>
</html>
`))

// httpGoroutine serves list of goroutines in a particular group.
func httpGoroutine(w http.ResponseWriter, r *http.Request) {
	// TODO(hyangah): support format=csv (raw data)

	events, err := parseTrace()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	pc, err := strconv.ParseUint(r.FormValue("id"), 10, 64)
	if err != nil {
		http.Error(w, fmt.Sprintf("failed to parse id parameter '%v': %v", r.FormValue("id"), err), http.StatusInternalServerError)
		return
	}
	analyzeGoroutines(events)
	var (
		glist                   []*trace.GDesc
		name                    string
		totalExecTime, execTime int64
		maxTotalTime            int64
	)

	for _, g := range gs {
		totalExecTime += g.ExecTime

		if g.PC != pc {
			continue
		}
		glist = append(glist, g)
		name = g.Name
		execTime += g.ExecTime
		if maxTotalTime < g.TotalTime {
			maxTotalTime = g.TotalTime
		}
	}

	execTimePercent := ""
	if totalExecTime > 0 {
		execTimePercent = fmt.Sprintf("%.2f%%", float64(execTime)/float64(totalExecTime)*100)
	}

	sortby := r.FormValue("sortby")
	_, ok := reflect.TypeOf(trace.GDesc{}).FieldByNameFunc(func(s string) bool {
		return s == sortby
	})
	if !ok {
		sortby = "TotalTime"
	}

	sort.Slice(glist, func(i, j int) bool {
		ival := reflect.ValueOf(glist[i]).Elem().FieldByName(sortby).Int()
		jval := reflect.ValueOf(glist[j]).Elem().FieldByName(sortby).Int()
		return ival > jval
	})

	err = templGoroutine.Execute(w, struct {
		Name            string
		PC              uint64
		N               int
		ExecTimePercent string
		MaxTotal        int64
		GList           []*trace.GDesc
	}{
		Name:            name,
		PC:              pc,
		N:               len(glist),
		ExecTimePercent: execTimePercent,
		MaxTotal:        maxTotalTime,
		GList:           glist})
	if err != nil {
		http.Error(w, fmt.Sprintf("failed to execute template: %v", err), http.StatusInternalServerError)
		return
	}
}

var templGoroutine = template.Must(template.New("").Funcs(template.FuncMap{
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
	"unknownTime": func(desc *trace.GDesc) int64 {
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

<table class="summary">
	<tr><td>Goroutine Name:</td><td>{{.Name}}</td></tr>
	<tr><td>Number of Goroutines:</td><td>{{.N}}</td></tr>
	<tr><td>Execution Time:</td><td>{{.ExecTimePercent}} of total program execution time </td> </tr>
	<tr><td>Network Wait Time:</td><td> <a href="/io?id={{.PC}}">graph</a><a href="/io?id={{.PC}}&raw=1" download="io.profile">(download)</a></td></tr>
	<tr><td>Sync Block Time:</td><td> <a href="/block?id={{.PC}}">graph</a><a href="/block?id={{.PC}}&raw=1" download="block.profile">(download)</a></td></tr>
	<tr><td>Blocking Syscall Time:</td><td> <a href="/syscall?id={{.PC}}">graph</a><a href="/syscall?id={{.PC}}&raw=1" download="syscall.profile">(download)</a></td></tr>
	<tr><td>Scheduler Wait Time:</td><td> <a href="/sched?id={{.PC}}">graph</a><a href="/sched?id={{.PC}}&raw=1" download="sched.profile">(download)</a></td></tr>
</table>
<p>
<table class="details">
<tr>
<th> Goroutine</th>
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
{{range .GList}}
  <tr>
    <td> <a href="/trace?goid={{.ID}}">{{.ID}}</a> </td>
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

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Goroutine-related profiles.

package main

import (
	"fmt"
	"html/template"
	"internal/trace"
	"net/http"
	"sort"
	"strconv"
	"sync"
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

type gtypeList []gtype

func (l gtypeList) Len() int {
	return len(l)
}

func (l gtypeList) Less(i, j int) bool {
	return l[i].ExecTime > l[j].ExecTime
}

func (l gtypeList) Swap(i, j int) {
	l[i], l[j] = l[j], l[i]
}

type gdescList []*trace.GDesc

func (l gdescList) Len() int {
	return len(l)
}

func (l gdescList) Less(i, j int) bool {
	return l[i].TotalTime > l[j].TotalTime
}

func (l gdescList) Swap(i, j int) {
	l[i], l[j] = l[j], l[i]
}

var (
	gsInit sync.Once
	gs     map[uint64]*trace.GDesc
)

// analyzeGoroutines generates statistics about execution of all goroutines and stores them in gs.
func analyzeGoroutines(events []*trace.Event) {
	gsInit.Do(func() {
		gs = trace.GoroutineStats(events)
	})
}

// httpGoroutines serves list of goroutine groups.
func httpGoroutines(w http.ResponseWriter, r *http.Request) {
	events, err := parseEvents()
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
	var glist gtypeList
	for k, v := range gss {
		v.ID = k
		glist = append(glist, v)
	}
	sort.Sort(glist)
	templGoroutines.Execute(w, glist)
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
	events, err := parseEvents()
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
	var glist gdescList
	for _, g := range gs {
		if g.PC != pc || g.ExecTime == 0 {
			continue
		}
		glist = append(glist, g)
	}
	sort.Sort(glist)
	err = templGoroutine.Execute(w, glist)
	if err != nil {
		http.Error(w, fmt.Sprintf("failed to execute template: %v", err), http.StatusInternalServerError)
		return
	}
}

var templGoroutine = template.Must(template.New("").Parse(`
<html>
<body>
<table border="1" sortable="1">
<tr>
<th> Goroutine </th>
<th> Total time, ns </th>
<th> Execution time, ns </th>
<th> Network wait time, ns </th>
<th> Sync block time, ns </th>
<th> Blocking syscall time, ns </th>
<th> Scheduler wait time, ns </th>
<th> GC sweeping time, ns </th>
<th> GC pause time, ns </th>
</tr>
{{range $}}
  <tr>
    <td> <a href="/trace?goid={{.ID}}">{{.ID}}</a> </td>
    <td> {{.TotalTime}} </td>
    <td> {{.ExecTime}} </td>
    <td> {{.IOTime}} </td>
    <td> {{.BlockTime}} </td>
    <td> {{.SyscallTime}} </td>
    <td> {{.SchedWaitTime}} </td>
    <td> {{.SweepTime}} </td>
    <td> {{.GCTime}} </td>
  </tr>
{{end}}
</table>
</body>
</html>
`))

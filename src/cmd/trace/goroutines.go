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

// gdesc desribes a single goroutine.
type gdesc struct {
	ID         uint64
	Name       string
	PC         uint64
	CreateTime int64
	StartTime  int64
	EndTime    int64
	LastStart  int64

	ExecTime      int64
	SchedWaitTime int64
	IOTime        int64
	BlockTime     int64
	SyscallTime   int64
	GCTime        int64
	SweepTime     int64
	TotalTime     int64

	blockNetTime     int64
	blockSyncTime    int64
	blockSyscallTime int64
	blockSweepTime   int64
	blockGCTime      int64
	blockSchedTime   int64
}

type gdescList []*gdesc

func (l gdescList) Len() int {
	return len(l)
}

func (l gdescList) Less(i, j int) bool {
	return l[i].TotalTime > l[j].TotalTime
}

func (l gdescList) Swap(i, j int) {
	l[i], l[j] = l[j], l[i]
}

var gs = make(map[uint64]*gdesc)

// analyzeGoroutines generates list gdesc's from the trace and stores it in gs.
func analyzeGoroutines(events []*trace.Event) {
	if len(gs) > 0 { //!!! racy
		return
	}
	var lastTs int64
	var gcStartTime int64
	for _, ev := range events {
		lastTs = ev.Ts
		switch ev.Type {
		case trace.EvGoCreate:
			g := &gdesc{CreateTime: ev.Ts}
			g.blockSchedTime = ev.Ts
			gs[ev.Args[0]] = g
		case trace.EvGoStart:
			g := gs[ev.G]
			if g.PC == 0 {
				g.PC = ev.Stk[0].PC
				g.Name = ev.Stk[0].Fn
			}
			g.LastStart = ev.Ts
			if g.StartTime == 0 {
				g.StartTime = ev.Ts
			}
			if g.blockSchedTime != 0 {
				g.SchedWaitTime += ev.Ts - g.blockSchedTime
				g.blockSchedTime = 0
			}
		case trace.EvGoEnd, trace.EvGoStop:
			g := gs[ev.G]
			g.ExecTime += ev.Ts - g.LastStart
			g.TotalTime = ev.Ts - g.CreateTime
			g.EndTime = ev.Ts
		case trace.EvGoBlockSend, trace.EvGoBlockRecv, trace.EvGoBlockSelect,
			trace.EvGoBlockSync, trace.EvGoBlockCond:
			g := gs[ev.G]
			g.ExecTime += ev.Ts - g.LastStart
			g.blockSyncTime = ev.Ts
		case trace.EvGoSched, trace.EvGoPreempt:
			g := gs[ev.G]
			g.ExecTime += ev.Ts - g.LastStart
			g.blockSchedTime = ev.Ts
		case trace.EvGoSleep, trace.EvGoBlock:
			g := gs[ev.G]
			g.ExecTime += ev.Ts - g.LastStart
		case trace.EvGoBlockNet:
			g := gs[ev.G]
			g.ExecTime += ev.Ts - g.LastStart
			g.blockNetTime = ev.Ts
		case trace.EvGoUnblock:
			g := gs[ev.Args[0]]
			if g.blockNetTime != 0 {
				g.IOTime += ev.Ts - g.blockNetTime
				g.blockNetTime = 0
			}
			if g.blockSyncTime != 0 {
				g.BlockTime += ev.Ts - g.blockSyncTime
				g.blockSyncTime = 0
			}
			g.blockSchedTime = ev.Ts
		case trace.EvGoSysBlock:
			g := gs[ev.G]
			g.ExecTime += ev.Ts - g.LastStart
			g.blockSyscallTime = ev.Ts
		case trace.EvGoSysExit:
			g := gs[ev.G]
			if g.blockSyscallTime != 0 {
				g.SyscallTime += ev.Ts - g.blockSyscallTime
				g.blockSyscallTime = 0
			}
			g.blockSchedTime = ev.Ts
		case trace.EvGCSweepStart:
			g := gs[ev.G]
			if g != nil {
				// Sweep can happen during GC on system goroutine.
				g.blockSweepTime = ev.Ts
			}
		case trace.EvGCSweepDone:
			g := gs[ev.G]
			if g != nil && g.blockSweepTime != 0 {
				g.SweepTime += ev.Ts - g.blockSweepTime
				g.blockSweepTime = 0
			}
		case trace.EvGCStart:
			gcStartTime = ev.Ts
		case trace.EvGCDone:
			for _, g := range gs {
				if g.EndTime == 0 {
					g.GCTime += ev.Ts - gcStartTime
				}
			}
		}
	}

	for _, g := range gs {
		if g.TotalTime == 0 {
			g.TotalTime = lastTs - g.CreateTime
		}
		if g.EndTime == 0 {
			g.EndTime = lastTs
		}
		if g.blockNetTime != 0 {
			g.IOTime += lastTs - g.blockNetTime
			g.blockNetTime = 0
		}
		if g.blockSyncTime != 0 {
			g.BlockTime += lastTs - g.blockSyncTime
			g.blockSyncTime = 0
		}
		if g.blockSyscallTime != 0 {
			g.SyscallTime += lastTs - g.blockSyscallTime
			g.blockSyscallTime = 0
		}
		if g.blockSchedTime != 0 {
			g.SchedWaitTime += lastTs - g.blockSchedTime
			g.blockSchedTime = 0
		}
	}
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
	for gid, g := range gs {
		if g.PC != pc || g.ExecTime == 0 {
			continue
		}
		g.ID = gid
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

// relatedGoroutines finds set of related goroutines that we need to include
// into trace for goroutine goid.
func relatedGoroutines(events []*trace.Event, goid uint64) map[uint64]bool {
	// BFS of depth 2 over "unblock" edges
	// (what goroutines unblock goroutine goid?).
	gmap := make(map[uint64]bool)
	gmap[goid] = true
	for i := 0; i < 2; i++ {
		gmap1 := make(map[uint64]bool)
		for g := range gmap {
			gmap1[g] = true
		}
		for _, ev := range events {
			if ev.Type == trace.EvGoUnblock && gmap[ev.Args[0]] {
				gmap1[ev.G] = true
			}
		}
		gmap = gmap1
	}
	gmap[0] = true // for GC events
	return gmap
}

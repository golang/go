// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trace

// GDesc contains statistics about execution of a single goroutine.
type GDesc struct {
	ID           uint64
	Name         string
	PC           uint64
	CreationTime int64
	StartTime    int64
	EndTime      int64

	ExecTime      int64
	SchedWaitTime int64
	IOTime        int64
	BlockTime     int64
	SyscallTime   int64
	GCTime        int64
	SweepTime     int64
	TotalTime     int64

	*gdesc // private part
}

// gdesc is a private part of GDesc that is required only during analysis.
type gdesc struct {
	lastStartTime    int64
	blockNetTime     int64
	blockSyncTime    int64
	blockSyscallTime int64
	blockSweepTime   int64
	blockGCTime      int64
	blockSchedTime   int64
}

// GoroutineStats generates statistics for all goroutines in the trace.
func GoroutineStats(events []*Event) map[uint64]*GDesc {
	gs := make(map[uint64]*GDesc)
	var lastTs int64
	var gcStartTime int64
	for _, ev := range events {
		lastTs = ev.Ts
		switch ev.Type {
		case EvGoCreate:
			g := &GDesc{ID: ev.Args[0], CreationTime: ev.Ts, gdesc: new(gdesc)}
			g.blockSchedTime = ev.Ts
			gs[g.ID] = g
		case EvGoStart, EvGoStartLabel:
			g := gs[ev.G]
			if g.PC == 0 {
				g.PC = ev.Stk[0].PC
				g.Name = ev.Stk[0].Fn
			}
			g.lastStartTime = ev.Ts
			if g.StartTime == 0 {
				g.StartTime = ev.Ts
			}
			if g.blockSchedTime != 0 {
				g.SchedWaitTime += ev.Ts - g.blockSchedTime
				g.blockSchedTime = 0
			}
		case EvGoEnd, EvGoStop:
			g := gs[ev.G]
			g.ExecTime += ev.Ts - g.lastStartTime
			g.TotalTime = ev.Ts - g.CreationTime
			g.EndTime = ev.Ts
		case EvGoBlockSend, EvGoBlockRecv, EvGoBlockSelect,
			EvGoBlockSync, EvGoBlockCond:
			g := gs[ev.G]
			g.ExecTime += ev.Ts - g.lastStartTime
			g.blockSyncTime = ev.Ts
		case EvGoSched, EvGoPreempt:
			g := gs[ev.G]
			g.ExecTime += ev.Ts - g.lastStartTime
			g.blockSchedTime = ev.Ts
		case EvGoSleep, EvGoBlock:
			g := gs[ev.G]
			g.ExecTime += ev.Ts - g.lastStartTime
		case EvGoBlockNet:
			g := gs[ev.G]
			g.ExecTime += ev.Ts - g.lastStartTime
			g.blockNetTime = ev.Ts
		case EvGoBlockGC:
			g := gs[ev.G]
			g.ExecTime += ev.Ts - g.lastStartTime
			g.blockGCTime = ev.Ts
		case EvGoUnblock:
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
		case EvGoSysBlock:
			g := gs[ev.G]
			g.ExecTime += ev.Ts - g.lastStartTime
			g.blockSyscallTime = ev.Ts
		case EvGoSysExit:
			g := gs[ev.G]
			if g.blockSyscallTime != 0 {
				g.SyscallTime += ev.Ts - g.blockSyscallTime
				g.blockSyscallTime = 0
			}
			g.blockSchedTime = ev.Ts
		case EvGCSweepStart:
			g := gs[ev.G]
			if g != nil {
				// Sweep can happen during GC on system goroutine.
				g.blockSweepTime = ev.Ts
			}
		case EvGCSweepDone:
			g := gs[ev.G]
			if g != nil && g.blockSweepTime != 0 {
				g.SweepTime += ev.Ts - g.blockSweepTime
				g.blockSweepTime = 0
			}
		case EvGCStart:
			gcStartTime = ev.Ts
		case EvGCDone:
			for _, g := range gs {
				if g.EndTime == 0 {
					g.GCTime += ev.Ts - gcStartTime
				}
			}
		}
	}

	for _, g := range gs {
		if g.TotalTime == 0 {
			g.TotalTime = lastTs - g.CreationTime
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
		g.gdesc = nil
	}

	return gs
}

// RelatedGoroutines finds a set of goroutines related to goroutine goid.
func RelatedGoroutines(events []*Event, goid uint64) map[uint64]bool {
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
			if ev.Type == EvGoUnblock && gmap[ev.Args[0]] {
				gmap1[ev.G] = true
			}
		}
		gmap = gmap1
	}
	gmap[0] = true // for GC events
	return gmap
}

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package traceparser

// postProcess is a final check of consistency, and if all is well,
// adds links to Events

import (
	"fmt"
)

type gStatus int

const (
	gDead gStatus = iota
	gRunnable
	gRunning
	gWaiting
)

// This code is copied from internal/trace/parser.go. With greater understanding it could be
// simplified. Sets ev.P for GCStart, and set various Link fields
func (p *Parsed) postProcess(events []*Event) error {
	type gdesc struct {
		state        gStatus
		ev           *Event
		evStart      *Event
		evCreate     *Event
		evMarkAssist *Event
	}
	type pdesc struct {
		running bool
		g       uint64
		evSTW   *Event
		evSweep *Event
	}

	gs := make(map[uint64]gdesc)
	ps := make(map[int]pdesc)
	tasks := make(map[uint64]*Event)           // task id to task creation events
	activeRegions := make(map[uint64][]*Event) // goroutine id to stack of spans
	gs[0] = gdesc{state: gRunning}
	var evGC, evSTW *Event

	checkRunning := func(pd pdesc, g gdesc, ev *Event, allowG0 bool) error {
		if g.state != gRunning {
			return fmt.Errorf("saw %v, but g %d is not running", ev, ev.G)
		}
		if pd.g != ev.G {
			return fmt.Errorf("saw %v, but it's P is running %d, not %d", ev, pd.g, ev.G)
		}
		if !allowG0 && ev.G == 0 {
			return fmt.Errorf("saw %v with unexpected g==0", ev)
		}
		return nil
	}
	for i, ev := range events {
		g := gs[ev.G]
		px := ps[int(ev.P)]
		switch ev.Type {
		case EvProcStart:
			if px.running {
				return fmt.Errorf("%d: running before start %s", i, ev)
			}
			px.running = true
		case EvProcStop:
			if !px.running {
				return fmt.Errorf("%d: p %d not running %s", i, ev.P, ev)
			}
			if px.g != 0 {
				return fmt.Errorf("p %d is running a goroutine %s", ev.P, ev)
			}
			px.running = false
		case EvGCStart:
			if evGC != nil {
				return fmt.Errorf("GC already running %s, was %s", ev, evGC)
			}
			evGC = ev
			// Attribute this to the global GC state.
			ev.P = GCP
		case EvGCDone:
			if evGC == nil {
				return fmt.Errorf("%d:%s bogus GC end", i, ev)
			}
			evGC.Link = ev
			evGC = nil
		case EvGCSTWStart:
			evp := &evSTW
			if p.Version < 1010 {
				// Before 1.10, EvGCSTWStart was per-P.
				evp = &px.evSTW
			}
			if *evp != nil {
				return fmt.Errorf("STW %s still running at %s", *evp, ev)
			}
			*evp = ev
		case EvGCSTWDone:
			evp := &evSTW
			if p.Version < 1010 {
				// Before 1.10, EvGCSTWDone was per-P.
				evp = &px.evSTW
			}
			if *evp == nil {
				return fmt.Errorf("%d: no STW running %s", i, ev)
			}
			(*evp).Link = ev
			*evp = nil
		case EvGCMarkAssistStart:
			if g.evMarkAssist != nil {
				return fmt.Errorf("%d: MarkAssist %s is still running at %s",
					i, g.evMarkAssist, ev)
			}
			g.evMarkAssist = ev
		case EvGCMarkAssistDone:
			// Unlike most events, mark assists can be in progress when a
			// goroutine starts tracing, so we can't report an error here.
			if g.evMarkAssist != nil {
				g.evMarkAssist.Link = ev
				g.evMarkAssist = nil
			}
		case EvGCSweepStart:
			if px.evSweep != nil {
				return fmt.Errorf("sweep not done %d: %s", i, ev)
			}
			px.evSweep = ev
		case EvGCSweepDone:
			if px.evSweep == nil {
				return fmt.Errorf("%d: no sweep happening %s", i, ev)
			}
			px.evSweep.Link = ev
			px.evSweep = nil
		case EvGoWaiting:
			if g.state != gRunnable {
				return fmt.Errorf("not runnable before %d:%s", i, ev)
			}
			g.state = gWaiting
			g.ev = ev
		case EvGoInSyscall:
			if g.state != gRunnable {
				return fmt.Errorf("not runnable before %d:%s", i, ev)
			}
			g.state = gWaiting
			g.ev = ev
		case EvGoCreate:
			if err := checkRunning(px, g, ev, true); err != nil {
				return err
			}
			if _, ok := gs[ev.Args[0]]; ok {
				return fmt.Errorf("%d: already exists %s", i, ev)
			}
			gs[ev.Args[0]] = gdesc{state: gRunnable, ev: ev, evCreate: ev}
		case EvGoStart, EvGoStartLabel:
			if g.state != gRunnable {
				return fmt.Errorf("not runnable before start %d:%s %+v", i, ev, g)
			}
			if px.g != 0 {
				return fmt.Errorf("%d: %s has p running %d already %v", i, ev, px.g, px)
			}
			g.state = gRunning
			g.evStart = ev
			px.g = ev.G
			if g.evCreate != nil {
				if p.Version < 1007 {
					// +1 because symbolizer expects return pc.
					//PJW: aren't doing < 1007. ev.stk = []*Frame{{PC: g.evCreate.args[1] + 1}}
				} else {
					ev.StkID = uint32(g.evCreate.Args[1])
				}
				g.evCreate = nil
			}

			if g.ev != nil {
				g.ev.Link = ev
				g.ev = nil
			}
		case EvGoEnd, EvGoStop:
			if err := checkRunning(px, g, ev, false); err != nil {
				return fmt.Errorf("%d: %s", i, err)
			}
			g.evStart.Link = ev
			g.evStart = nil
			g.state = gDead
			px.g = 0

			if ev.Type == EvGoEnd { // flush all active Regions
				spans := activeRegions[ev.G]
				for _, s := range spans {
					s.Link = ev
				}
				delete(activeRegions, ev.G)
			}
		case EvGoSched, EvGoPreempt:
			if err := checkRunning(px, g, ev, false); err != nil {
				return err
			}
			g.state = gRunnable
			g.evStart.Link = ev
			g.evStart = nil
			px.g = 0
			g.ev = ev
		case EvGoUnblock:
			if g.state != gRunning { // PJW, error message
				return fmt.Errorf("Event %d (%s) is not running at unblock %s", i, ev, g.state)

			}
			if ev.P != TimerP && px.g != ev.G {
				// PJW: do better here.
				return fmt.Errorf("%d: %s p %d is not running g", i, ev, px.g)
			}
			g1 := gs[ev.Args[0]]
			if g1.state != gWaiting {
				return fmt.Errorf("g %v is not waiting before unpark i=%d g1=%v %s",
					ev.Args[0], i, g1, ev)
			}
			if g1.ev != nil && g1.ev.Type == EvGoBlockNet && ev.P != TimerP {
				ev.P = NetpollP
			}
			if g1.ev != nil {
				g1.ev.Link = ev
			}
			g1.state = gRunnable
			g1.ev = ev
			gs[ev.Args[0]] = g1
		case EvGoSysCall:
			if err := checkRunning(px, g, ev, false); err != nil {
				return err
			}
			g.ev = ev
		case EvGoSysBlock:
			if err := checkRunning(px, g, ev, false); err != nil {
				return err
			}
			g.state = gWaiting
			g.evStart.Link = ev
			g.evStart = nil
			px.g = 0
		case EvGoSysExit:
			if g.state != gWaiting {
				return fmt.Errorf("not waiting when %s", ev)
			}
			if g.ev != nil && g.ev.Type == EvGoSysCall {
				g.ev.Link = ev
			}
			g.state = gRunnable
			g.ev = ev
		case EvGoSleep, EvGoBlock, EvGoBlockSend, EvGoBlockRecv,
			EvGoBlockSelect, EvGoBlockSync, EvGoBlockCond, EvGoBlockNet, EvGoBlockGC:
			if err := checkRunning(px, g, ev, false); err != nil {
				return err
			}
			g.state = gWaiting
			g.ev = ev
			g.evStart.Link = ev
			g.evStart = nil
			px.g = 0
		case EvUserTaskCreate:
			taskid := ev.Args[0]
			if prevEv, ok := tasks[taskid]; ok {
				return fmt.Errorf("task id conflicts (id:%d), %q vs %q", taskid, ev, prevEv)
			}
			tasks[ev.Args[0]] = ev
		case EvUserTaskEnd:
			if prevEv, ok := tasks[ev.Args[0]]; ok {
				prevEv.Link = ev
				ev.Link = prevEv
			}
		case EvUserRegion:
			mode := ev.Args[1]
			spans := activeRegions[ev.G]
			if mode == 0 { // span start
				activeRegions[ev.G] = append(spans, ev) // push
			} else if mode == 1 { // span end
				n := len(spans)
				if n > 0 { // matching span start event is in the trace.
					s := spans[n-1]
					if s.Args[0] != ev.Args[0] || s.SArgs[0] != ev.SArgs[0] { // task id, span name mismatch
						return fmt.Errorf("misuse of span in goroutine %d: span end %q when the inner-most active span start event is %q",
							ev.G, ev, s)
					}
					// Link span start event with span end event
					s.Link = ev
					ev.Link = s

					if n > 1 {
						activeRegions[ev.G] = spans[:n-1]
					} else {
						delete(activeRegions, ev.G)
					}
				}
			} else {
				return fmt.Errorf("invalid user region, mode: %q", ev)
			}
		}
		gs[ev.G] = g
		ps[int(ev.P)] = px
	}
	return nil
}
func (g gStatus) String() string {
	switch g {
	case gDead:
		return "gDead"
	case gRunnable:
		return "gRunnable"
	case gRunning:
		return "gRunning"
	case gWaiting:
		return "gWaiting"
	}
	return fmt.Sprintf("gStatus?%d", g)
}

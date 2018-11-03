// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package traceparser

import (
	"fmt"
	"sort"
)

// convert raw events into Events

func (p *Parsed) createEvents(f func(string)) error {
	// multiple passes:
	// create some Events
	// sort them by time (and adjust their times to be nanonseconds)
	// remove events not in the desired time interval
	// make the events consistent (adding initializing events at the beginning)
	// remove the futile events
	// add links (and do final checking)

	// shared by firstEvents
	p.byproc = make(map[int][]*Event)
	p.lastGs = make(map[int]uint64)

	// p.batches are always sorted by time. otherwise a batch for one p that is totally
	// later than another batch might be done first, confusing us about g's
	for i, b := range p.batches {
		if b.raws == nil {
			continue
		}
		if err := p.firstEvents(b); err != nil {
			return fmt.Errorf("%v", err) // PJW: this is not useful
		}
		// we done with b.raws now
		p.batches[i].raws = nil
	}
	f("firstEvents finished")
	sorted := []*Event{}
	for _, v := range p.byproc {
		sorted = append(sorted, v...)
	}
	// PJW: are we done with p.byproc now? Yes. This shrinks a little.
	p.byproc = nil
	// Why wasn't this done earlier? Or, why do it at all?
	for _, ev := range sorted {
		switch ev.Type {
		case EvGoStartLocal:
			ev.Type = EvGoStart
		case EvGoUnblockLocal:
			ev.Type = EvGoUnblock
		case EvGoSysExitLocal:
			ev.Type = EvGoSysExit
		}
	}
	// change to nanoseconds
	freq := 1e9 / float64(p.TicksPerSec)
	for i, ev := range sorted {
		// Move timers and syscalls to separate fake Ps.
		// This could be done in the loop at line 38
		// or maybe after robust fixes things.
		if p.timerGoids[ev.G] && ev.Type == EvGoUnblock {
			ev.Args[2] = uint64(ev.P) // save for robust() to use
			ev.P = TimerP
		}
		// sometimes the ts is not what it should be
		if ev.Type == EvGoSysExit {
			ev.P = SyscallP
			if ev.Args[2] != 0 {
				// PJW: test for this being safe. There might be no preceding
				// EvSysBlock, EvGoInSyscall, or its time might be later than this
				ev.Ts = int64(ev.Args[2])
			}
		}
		if ev.Type == EvGCStart {
			ev.P = GCP
		}
		t := ev.Ts - p.minticks
		if t < 0 {
			return fmt.Errorf("event %d %s would be %d mints=%x", i, ev, t, p.minticks)
		}
		ev.Ts = int64(float64(ev.Ts-p.minticks) * freq)
	}
	// Stable for the case of equal Ts's.
	sort.SliceStable(sorted, func(i, j int) bool { return sorted[i].Ts < sorted[j].Ts })
	f("sorted")
	// and ignore the ones with times out of bounds
	firstwant, lastwant := 0, len(sorted)
	for i, ev := range sorted {
		if ev.Ts < p.MinWant {
			firstwant = i + 1
		} else if ev.Ts > p.MaxWant { // closed interval [minwant, maxwant]
			lastwant = i
			break // sorted by Ts
		}
	}
	f("nanoseconds")
	var err error
	sorted, _, err = p.robust(sorted[firstwant:lastwant]) // PJW: copy info from aux
	f("consistent")
	if err != nil {
		return err
	}
	events, cnt := p.removeFutile(sorted) // err is always nil here.
	f(fmt.Sprintf("removed %d futiles", cnt))
	// and finally, do some checks and put in links
	err = p.postProcess(events)
	f("post processed")
	if err != nil {
		return err // PJW: is this enough? NO
	}
	p.Events = events
	return nil
}

// Special P identifiers.
const (
	FakeP    = 1000000 + iota
	TimerP   // depicts timer unblocks
	NetpollP // depicts network unblocks
	SyscallP // depicts returns from syscalls
	GCP      // depicts GC state
)

// convert the raw events for a batch into Events, and keep track of
// which G is running on the P that is common to the batch.
func (p *Parsed) firstEvents(b batch) error {
	for _, raw := range b.raws {
		desc := EventDescriptions[raw.typ]
		narg := p.rawArgNum(&raw)
		if p.Err != nil {
			return p.Err
		}
		if raw.typ == EvBatch {
			// first event, record information about P, G, and Ts
			p.lastGs[p.lastP] = p.lastG // 0 the first time through
			p.lastP = int(raw.Arg(0))
			p.lastG = p.lastGs[p.lastP]
			p.lastTs = int64(raw.Arg(1))
			continue
		}
		e := &Event{Type: raw.typ, P: int32(p.lastP), G: p.lastG}
		var argoffset int
		if p.Version < 1007 { // can't happen.
			e.Ts = p.lastTs + int64(raw.Arg(1))
			argoffset = 2
		} else {
			e.Ts = p.lastTs + int64(raw.Arg(0))
			argoffset = 1
		}
		p.lastTs = e.Ts
		// collect the args for the raw event e
		for i := argoffset; i < narg; i++ {
			// evade one byte of corruption (from fuzzing typically)
			if raw.args == nil {
				return fmt.Errorf("raw.args is nil %s", evname(raw.typ))
			}
			if i > 0 && i-1 >= len(*raw.args) {
				return fmt.Errorf("%s wants arg %d of %d", evname(raw.typ), i, len(*raw.args))
			}
			if i == narg-1 && desc.Stack {
				e.StkID = uint32(raw.Arg(i))
			} else {
				e.Args[i-argoffset] = raw.Arg(i)
			}
		}
		switch raw.typ {
		case EvGoSysCall, EvGCSweepDone, EvGCSweepStart:
			if e.G == 0 {
				// missing some earlier G's from this P
				continue // so we don't know which G is doing it
			}
		case EvGoStart, EvGoStartLocal, EvGoStartLabel:
			p.lastG = e.Args[0]
			e.G = p.lastG
			if raw.typ == EvGoStartLabel {
				e.SArgs = []string{p.Strings[e.Args[2]]}
			}
		case EvGCSTWStart:
			e.G = 0
			switch e.Args[0] {
			case 0:
				e.SArgs = []string{"mark termination"}
			case 1:
				e.SArgs = []string{"sweep termination"}
			default:
				return fmt.Errorf("unknown STW kind %d!=0,1 %s", e.Args[0], e)
			}
		case EvGCStart, EvGCDone, EvGCSTWDone:
			e.G = 0
		case EvGoEnd, EvGoStop, EvGoSched, EvGoPreempt,
			EvGoSleep, EvGoBlock, EvGoBlockSend, EvGoBlockRecv,
			EvGoBlockSelect, EvGoBlockSync, EvGoBlockCond, EvGoBlockNet,
			EvGoSysBlock, EvGoBlockGC:
			p.lastG = 0
			if e.G == 0 {
				// missing some earlier G's from this P
				continue // so we don't know which G is doing it
			}
		case EvGoSysExit, EvGoWaiting, EvGoInSyscall:
			e.G = e.Args[0]
		case EvUserTaskCreate:
			// e.Args 0: taskID, 1:parentID, 2:nameID
			e.SArgs = []string{p.Strings[e.Args[2]]}
		case EvUserRegion:
			if e.G == 0 {
				continue // don't know its G
			}
			// e.Args 0: taskID, 1: mode, 2:nameID
			e.SArgs = []string{p.Strings[e.Args[2]]}
		case EvUserLog:
			// e.Args 0: taskID, 1:keyID, 2: stackID
			e.SArgs = []string{p.Strings[e.Args[1]], raw.sarg}
		}
		p.byproc[p.lastP] = append(p.byproc[p.lastP], e)
	}
	return nil
}

func (p *Parsed) removeFutile(events []*Event) ([]*Event, int) {
	// Phase 1: determine futile wakeup sequences.
	type G struct {
		futile bool
		wakeup []*Event // wakeup sequence (subject for removal)
	}
	gs := make(map[uint64]G)
	futile := make(map[*Event]bool)
	cnt := 0
	for _, ev := range events {
		switch ev.Type {
		case EvGoUnblock:
			g := gs[ev.Args[0]]
			g.wakeup = []*Event{ev}
			gs[ev.Args[0]] = g
		case EvGoStart, EvGoPreempt, EvFutileWakeup:
			g := gs[ev.G]
			g.wakeup = append(g.wakeup, ev)
			if ev.Type == EvFutileWakeup {
				g.futile = true
			}
			gs[ev.G] = g
		case EvGoBlock, EvGoBlockSend, EvGoBlockRecv, EvGoBlockSelect,
			EvGoBlockSync, EvGoBlockCond:
			g := gs[ev.G]
			if g.futile {
				futile[ev] = true
				for _, ev1 := range g.wakeup {
					futile[ev1] = true
				}
			}
			delete(gs, ev.G)
			cnt++
		}
	}
	// Phase 2: remove futile wakeup sequences.
	newEvents := events[:0] // overwrite the original slice
	for _, ev := range events {
		if !futile[ev] {
			newEvents = append(newEvents, ev)
		}
	}
	return newEvents, cnt // PJW: cnt doesn't count the futile[]s
}

// Arg gets the n-th arg from a raw event
func (r *rawEvent) Arg(n int) uint64 {
	if n == 0 {
		return r.arg0
	}
	return (*r.args)[n-1]
}

// report the number of arguments. (historical differences)
func (p *Parsed) rawArgNum(ev *rawEvent) int {
	desc := EventDescriptions[ev.typ]
	switch ev.typ {
	case EvStack, EvFrequency, EvTimerGoroutine:
		p.Err = fmt.Errorf("%s unexpected in rawArgNum", evname(ev.typ))
		return 0
	}
	narg := len(desc.Args)
	if desc.Stack {
		narg++
	}
	if ev.typ == EvBatch {
		if p.Version < 1007 {
			narg++ // used to be an extra unused arg
		}
		return narg
	}
	narg++ // timestamp
	if p.Version < 1007 {
		narg++ // sequence
	}
	// various special historical cases
	switch ev.typ {
	case EvGCSweepDone:
		if p.Version < 1009 {
			narg -= 2 // 1.9 added 2 args
		}
	case EvGCStart, EvGoStart, EvGoUnblock:
		if p.Version < 1007 {
			narg-- // one more since 1.7
		}
	case EvGCSTWStart:
		if p.Version < 1010 {
			narg-- // 1.10 added an argument
		}
	}
	return narg
}

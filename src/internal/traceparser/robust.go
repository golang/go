// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package traceparser

// there are panics for impossible situations. probably an error would be better
// (if only it were certain these are impossible)

import (
	"fmt"
	"log"
)

// repair an incomplete or possibly damaged interval of Events
// so that postProcess is happy

// errors returned by checkRunning()
const (
	ok         = 0
	badRunning = 1 << iota
	badP
	badG0
)

// states of g's and p's
type gdesc struct {
	state                               gStatus
	ev, evStart, evCreate, evMarkAssist *Event
}

type pdesc struct {
	running        bool
	g              uint64
	evSTW, evSweep *Event
}

func locstr(ev *Event) string {
	if ev == nil {
		return "<nil>"
	}
	return fmt.Sprintf("%s:%x", evname(ev.Type), ev.Ts)
}
func (p pdesc) String() string {
	return fmt.Sprintf("[%v %d %s %s]", p.running, p.g, locstr(p.evSTW), locstr(p.evSweep))
}

func (g gdesc) String() string {
	var nm string
	switch g.state {
	case gDead:
		nm = "dead"
	case gWaiting:
		nm = "waiting"
	case gRunnable:
		nm = "runnable"
	case gRunning:
		nm = "running"
	}
	f := locstr
	return fmt.Sprintf("[%s %s,%s,%s,%s]", nm, f(g.ev), f(g.evStart),
		f(g.evCreate), f(g.evMarkAssist))
}

func checkRunning(pd pdesc, gd gdesc, ev *Event, okG0 bool) int {
	ret := ok
	if gd.state != gRunning {
		ret |= badRunning
	}
	if pd.g != ev.G {
		ret |= badP
	}
	if !okG0 && ev.G == 0 {
		ret |= badG0
	}
	return ret
}

type aux struct {
	pref     []*Event     // prefix
	evs      []*Event     // copies and inserted
	deleted  map[byte]int // count by Type
	inserted map[byte]int // count by Type
	gs       map[uint64]gdesc
	ps       map[int32]pdesc
	g        gdesc
	px       pdesc
	my       *Parsed
	input    []*Event // events in call to robust()
	last     int      // last index handled by reorder
	err      error    // report inconsistent trace files
}

func (a *aux) preftime() int64 {
	ts := a.my.MinWant - 1000
	if ts < 0 {
		ts = 0
	}
	if len(a.pref) > 0 {
		ts = a.pref[len(a.pref)-1].Ts + 1
	}
	return ts
}
func (a *aux) delete(i int, ev *Event) {
	a.deleted[ev.Type]++
}
func (a *aux) prefix(typ byte, g uint64, p int32) {
	ts := a.preftime()
	ev := &Event{Type: typ, G: g, P: p, Ts: ts,
		Args: [3]uint64{0, 0, 1}}
	a.pref = append(a.pref, ev)
}
func (a *aux) procstart(p int32) {
	if p >= FakeP || a.px.running {
		return
	}
	a.prefix(EvProcStart, 0, p)
	a.px.running = true
}
func (a *aux) makewaiting(i int, g uint64, typ byte) {
	// GoCreate, g=0 args[0]=g; maybe it exists already?
	// GoWaiting or GoInSysCall
	p := int32(a.my.batches[0].P)
	ev := &Event{Type: EvGoCreate, P: p,
		Ts: a.preftime(), Args: [3]uint64{g, 0, 2}}
	a.pref = append(a.pref, ev)
	a.gs[g] = gdesc{state: gRunnable, ev: ev, evCreate: ev}
	ev = &Event{Type: typ, G: g, P: p,
		Ts: a.preftime(), Args: [3]uint64{g, 0, 3}}
	a.pref = append(a.pref, ev)
	switch typ {
	default:
		panic(fmt.Sprintf("weird typ %s in makewaiting", evname(typ)))
	case EvGoWaiting, EvGoInSyscall:
		// ok
	}
}

func (a *aux) makerunnable(i int, ev *Event) {
	// Create, Sched, Preempt, or Unblock
	switch a.gs[ev.G].state {
	case gDead:
		p := int32(a.my.batches[0].P)
		ev := &Event{Type: EvGoCreate, P: p,
			Ts: a.preftime(), Args: [3]uint64{ev.G, 0, 4}}
		a.pref = append(a.pref, ev)
		a.gs[ev.Args[0]] = gdesc{state: gRunnable, ev: ev, evCreate: ev}
	case gRunnable:
		return
	case gRunning:
		//a.prevs(i)
		a.err = fmt.Errorf("gRunning %d:%s", i, ev)
	case gWaiting:
		//a.prevs(i)
		a.err = fmt.Errorf("no consistent ordering possible %d:%s", i, ev)
	}
}
func (a *aux) makerunning(i int, ev *Event) {
	// GoStart once it is runnable
	switch a.g.state {
	case gDead:
		a.makerunnable(i, ev)
	case gRunnable:
		break
	case gRunning:
		return
	case gWaiting:
		a.err = fmt.Errorf("gWaiting in makerunnable %d:%s %+v", i, ev, a.g)
	}
	// PJW: which P? Probably need a ProcStart once
	if !a.px.running {
		a.procstart(ev.P)
	}
	p := ev.P
	if p == TimerP {
		p = int32(ev.Args[2]) // from events.go:71
		ev.Args[2] = 0
	}
	x := &Event{Type: EvGoStart, G: ev.G, P: p, Args: [3]uint64{ev.G, 0, 5}}
	x.Ts = ev.Ts - 1
	a.evs = append(a.evs, x)
	a.g.state = gRunning
	a.g.evStart = x
	a.px.g = x.G
	a.inserted[EvGoStart]++
}

func (p *Parsed) robust(events []*Event) ([]*Event, *aux, error) { // *aux for debugging (CheckRobust)
	a := new(aux)
	a.gs = make(map[uint64]gdesc)
	a.ps = make(map[int32]pdesc)
	var evGC, evSTW *Event
	tasks := make(map[uint64]*Event) // task id to create
	activeSpans := make(map[uint64][]*Event)
	a.gs[0] = gdesc{state: gRunning} // bootstrap
	a.deleted = make(map[byte]int)
	a.inserted = make(map[byte]int)
	a.my = p
	a.input = events

	for i, ev := range events {
		if a.err != nil {
			break
		}
		if i < len(events)-1 && ev.Ts == events[i+1].Ts &&
			i > a.last {
			// sigh. dragonfly, or similar trouble.
			// a.last is to avoid overlapping calls
			// This is a placeholder if needed.
			//a.reorder(i, events)
			ev = events[i]
		}
		var gok, pok bool
		a.g, gok = a.gs[ev.G]
		a.px, pok = a.ps[ev.P]
		switch ev.Type {
		case EvProcStart:
			if a.px.running { // This doesn't happen, but to be safe
				a.delete(i, ev) // already started
				continue
			}
			a.px.running = true
		case EvProcStop:
			if !pok { // Ok to delete, as we've never heard of it
				a.delete(i, ev)
				continue
			}
			if !a.px.running {
				a.procstart(ev.P)
			}
			if a.px.g != 0 {
				// p is running a g! Stop the g? Ignore the Stop?
				// Ignore the Stop. I don't think this happens.
				// (unless there are equal Ts's or the file is corrupt)
				a.err = fmt.Errorf("unexpected %d:%s %v", i, ev, a.px)
				// a.delete(i, ev) // PJW
				continue
			}
			a.px.running = false
		case EvGCStart:
			if evGC != nil {
				// already running; doesn't happen
				a.delete(i, ev)
				continue
			}
			evGC = ev
		case EvGCDone:
			if evGC == nil {
				// no GCStart to link it to: choice is lying about
				// the duration or the existence. Do the latter
				a.delete(i, ev)
				continue
			}
			evGC = nil
		case EvGCSTWStart:
			evp := &evSTW
			if p.Version < 1010 {
				// Before 1.10, EvGCSTWStart was per-P.
				evp = &a.px.evSTW
			}
			if *evp != nil {
				// still running; doesn't happen
				a.delete(i, ev)
				continue
			}
			*evp = ev
		case EvGCSTWDone:
			evp := &evSTW
			if p.Version < 1010 {
				// Before 1.10, EvGCSTWDone was per-P.
				evp = &a.px.evSTW
			}
			if *evp == nil {
				// no STWStart to link to: choice is lying about
				// duration or the existence. Do the latter.
				a.delete(i, ev)
				continue
			}
			*evp = nil
		case EvGCMarkAssistStart:
			if a.g.evMarkAssist != nil {
				// already running; doesn't happen
				a.delete(i, ev)
				continue
			}
			a.g.evMarkAssist = ev
		case EvGCMarkAssistDone:
			// ok to be in progress
			a.g.evMarkAssist = nil
		case EvGCSweepStart:
			if a.px.evSweep != nil {
				// older one still running; doesn't happen
				a.delete(i, ev)
				continue
			}
			a.px.evSweep = ev
		case EvGCSweepDone:
			if a.px.evSweep == nil {
				// no Start to link to: choice is lying about
				// duration or existence. Do the latter.
				a.delete(i, ev)
				continue
			}
			a.px.evSweep = nil
		case EvGoWaiting:
			if a.g.state != gRunnable {
				a.makerunnable(i, ev)
			}
			a.g.state = gWaiting
			a.g.ev = ev
		case EvGoInSyscall: // PJW: same as GoWaiting
			if a.g.state != gRunnable {
				a.makerunnable(i, ev)
			}
			a.g.state = gWaiting
			a.g.ev = ev
		case EvGoCreate:
			if _, ok := a.gs[ev.Args[0]]; ok {
				// this g already exists; doesn't happen
				a.delete(i, ev)
				continue
			}
			ret := checkRunning(a.px, a.g, ev, true)
			if ret&badRunning != 0 {
				a.makerunning(i, ev)
				a.g.state = gRunning
			}
			if ret&badP != 0 {
				a.procstart(ev.P)
			}
			a.gs[ev.Args[0]] = gdesc{state: gRunnable, ev: ev,
				evCreate: ev}
		case EvGoStart, EvGoStartLabel:
			if a.g.state != gRunnable {
				a.makerunnable(i, ev)
			}
			if a.px.g != 0 {
				//a.prevs(i)
				a.err = fmt.Errorf("p already running %d, %d:%s",
					a.px.g, i, ev)
			}
			a.g.state = gRunning
			a.g.evStart = ev // PJW: do we need g.evStart?
			a.px.g = ev.G
			a.g.evCreate = nil // PJW: do we need g.evCreate?
		case EvGoEnd, EvGoStop:
			if !gok {
				// never heard of it; act as if it never existed
				a.delete(i, ev)
				continue
			}
			ret := checkRunning(a.px, a.g, ev, false)
			if ret&badRunning != 0 {
				a.makerunning(i, ev)
				a.g.state = gRunning
			}
			if ret&badP != 0 {
				a.procstart(ev.P)
			}
			if ret&badG0 != 0 {
				// gok should have been false
				panic(fmt.Sprintf("badG0 %d:%s", i, ev))
			}
			a.g.evStart = nil
			a.g.state = gDead
			a.px.g = 0
		case EvGoSched, EvGoPreempt:
			ret := checkRunning(a.px, a.g, ev, false)
			if ret&badG0 != 0 {
				// hopeless, we think. Don't know g
				a.delete(i, ev)
			}
			if ret&badRunning != 0 {
				a.makerunning(i, ev)
				a.g.state = gRunning
			}
			if ret&badP != 0 {
				a.procstart(ev.P)
			}
			a.g.state = gRunnable
			a.g.evStart = nil
			a.px.g = 0
			a.g.ev = ev
		case EvGoUnblock:
			// g == 0 is ok here (PJW) and elsewhere?
			if a.g.state != gRunning {
				a.makerunning(i, ev)
				a.g.state = gRunning
			}
			if ev.P != TimerP && a.px.g != ev.G {
				//a.prevs(i)
				a.err = fmt.Errorf("%v not running %d:%s",
					a.px, i, ev)
				continue
			}
			g1, _ := a.gs[ev.Args[0]]
			if g1.state != gWaiting {
				a.makewaiting(i, ev.Args[0], EvGoWaiting)
				g1.state = gWaiting
			}
			g1.state = gRunnable
			g1.ev = ev
			a.gs[ev.Args[0]] = g1
			// if p == TimerP, clean up from events.go:71
			ev.Args[2] = 0 // no point in checking p
		case EvGoSysCall:
			if ev.G == 0 {
				// hopeless; don't know how to repair
				a.delete(i, ev)
				continue
			}
			ret := checkRunning(a.px, a.g, ev, false)
			if ret&badRunning != 0 {
				a.makerunning(i, ev)
				a.g.state = gRunning
			}
			if ret&badP != 0 {
				a.procstart(ev.P)
			}
			a.g.ev = ev
		case EvGoSysBlock:
			if ev.G == 0 {
				// hopeless to repair
				a.delete(i, ev)
			}
			ret := checkRunning(a.px, a.g, ev, false)
			if ret&badRunning != 0 {
				a.makerunning(i, ev)
				a.g.state = gRunning
			}
			if ret&badP != 0 {
				a.procstart(ev.P)
			}
			a.g.state = gWaiting
			a.g.evStart = nil
			a.px.g = 0
		case EvGoSysExit:
			if ev.G == 0 {
				// don't know how to repair
				a.delete(i, ev)
				continue
			}
			if a.g.state != gWaiting {
				a.makewaiting(i, ev.G, EvGoInSyscall)
			}
			a.g.state = gRunnable
			a.g.ev = ev
		case EvGoSleep, EvGoBlock, EvGoBlockSend, EvGoBlockRecv,
			EvGoBlockSelect, EvGoBlockSync, EvGoBlockCond,
			EvGoBlockNet, EvGoBlockGC:
			if ev.G == 0 { // don't know how to repair
				a.delete(i, ev)
				continue
			}
			ret := checkRunning(a.px, a.g, ev, false)
			if ret&badRunning != 0 {
				a.makerunning(i, ev)
				a.g.state = gRunning
			}
			if ret&badP != 0 {
				a.procstart(ev.P)
			}
			a.g.state = gWaiting
			a.g.ev = ev
			a.g.evStart = nil
			a.px.g = 0
		case EvHeapAlloc, EvGomaxprocs, EvNextGC, EvUserLog:
			a.makerunning(i, ev)
			a.g.state = gRunning
			a.px.g = ev.G
		default:
			return nil, nil, fmt.Errorf("robust: unexpected %d:%s", i, ev)
		case EvUserTaskCreate:
			taskid := ev.Args[0]
			if _, ok := tasks[taskid]; ok {
				// task id conflict, kill this one, believe the earlier one
				a.delete(i, ev)
				continue
			}
			tasks[ev.Args[0]] = ev
		case EvUserTaskEnd: // nothing to do
		case EvUserRegion:
			mode := ev.Args[1]
			spans := activeSpans[ev.G]
			if mode == 0 {
				activeSpans[ev.G] = append(spans, ev)
			} else if mode == 1 { // span end
				n := len(spans)
				if n > 0 {
					// check that spans match up; clean up if not
					s := spans[n-1]
					if s.Args[0] != ev.Args[0] ||
						s.SArgs[0] != ev.SArgs[0] {
						// try to fix it
						var ok bool
						spans, ok = fixSpan(spans, ev)
						if !ok {
							// unfixed, toss this event
							a.delete(i, ev)
							continue
						}
					}
					n = len(spans)
					if n > 1 {
						activeSpans[ev.G] = spans[:n-1]
					} else {
						delete(activeSpans, ev.G)
					}
				}
			} else {
				// invalid mode, toss it
				a.delete(i, ev)
				continue
			}
		}
		a.gs[ev.G] = a.g
		a.ps[ev.P] = a.px
		a.evs = append(a.evs, ev)
	}
	ans := a.pref
	ans = append(ans, a.evs...)
	p.Preflen = len(a.pref)
	p.Added = len(a.inserted)
	p.Ignored = len(a.deleted)
	return ans, a, a.err
}

func fixSpan(spans []*Event, ev *Event) ([]*Event, bool) {
	// probably indicates a corrupt trace file
	panic("implement")
}

type same struct {
	ev *Event
	g  gdesc
	p  pdesc
}

// This is a placeholder, to organize intervals with equal time stamps
func (a *aux) reorder(n int, events []*Event) {
	// bunch of Events with equal time stamps
	// We care about GoCreate, GoWaiting, GoInSyscall,
	// GoStart (StartLocal, StartLabel), GoBlock*,
	// GosSched, GoPreempt, GoUnblock, GoSysExit,
	// (UnblockLocal, SysExitLocal), GCStart.
	// maybe ProcStart and ProcStop?
	repair := []same{}
	i := n
	for ; i < len(events) && events[i].Ts == events[n].Ts; i++ {
		ev := events[i]
		repair = append(repair, same{ev, a.gs[ev.G],
			a.ps[ev.P]})
	}
	a.last = i - 1
	log.Println("BEFORE:")
	for i, r := range repair {
		log.Printf("x%d:%s %v %v", i+n, r.ev, r.g, r.p)
	}
	if true { // PJW
		return // we're not doing anything yet
	}
	// sorting is not going to be enough.
	log.Println("DID NOTHING!")
	log.Println("after")
	for i, r := range repair {
		log.Printf("y%d:%s %v %v", i+n, r.ev, r.g, r.p)
	}
	for i, r := range repair {
		events[n+i] = r.ev
	}
}

// printing for debugging
func (a *aux) prevs(n int) {
	for i := 0; i < len(a.pref); i++ {
		log.Printf("p%3d %s", i, a.pref[i])
	}
	start := 0
	if n > 50 {
		start = n - 50
	}
	for i := start; i <= n+1 && i < len(a.input); i++ {
		log.Printf("%4d %s", i, a.input[i])
	}
}

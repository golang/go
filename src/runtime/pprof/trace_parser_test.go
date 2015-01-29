// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pprof_test

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"os/exec"
	"sort"
	"strconv"
	"strings"
)

// Event describes one event in the trace.
type Event struct {
	off   int       // offset in input file (for debugging and error reporting)
	typ   byte      // one of traceEv*
	ts    int64     // timestamp in nanoseconds
	p     int       // P on which the event happened (can be one of timerP, netpollP, syscallP)
	g     uint64    // G on which the event happened
	stkID uint64    // unique stack ID
	stk   []*Frame  // stack trace (can be empty)
	args  [2]uint64 // event-type-specific arguments
	// linked event (can be nil), depends on event type:
	// for GCStart: the GCStop
	// for GCScanStart: the GCScanDone
	// for GCSweepStart: the GCSweepDone
	// for GoCreate: first GoStart of the created goroutine
	// for GoStart: the associated GoEnd, GoBlock or other blocking event
	// for GoSched/GoPreempt: the next GoStart
	// for GoBlock and other blocking events: the unblock event
	// for GoUnblock: the associated GoStart
	// for blocking GoSysCall: the associated GoSysExit
	// for GoSysExit: the next GoStart
	link *Event
}

// Frame is a frame in stack traces.
type Frame struct {
	pc   uint64
	fn   string
	file string
	line int
}

const (
	// Special P identifiers:
	timerP   = 1000000 + iota // depicts timer unblocks
	netpollP                  // depicts network unblocks
	syscallP                  // depicts returns from syscalls
)

// parseTrace parses, post-processes and verifies the trace.
func parseTrace(r io.Reader) ([]*Event, error) {
	rawEvents, err := readTrace(r)
	if err != nil {
		return nil, err
	}
	events, err := parseEvents(rawEvents)
	if err != nil {
		return nil, err
	}
	err = postProcessTrace(events)
	if err != nil {
		return nil, err
	}
	return events, nil
}

// RawEvent is a helper type used during parsing.
type RawEvent struct {
	off  int
	typ  byte
	args []uint64
}

// readTrace does wire-format parsing and verification.
// It does not care about specific event types and argument meaning.
func readTrace(r io.Reader) ([]RawEvent, error) {
	// Read and validate trace header.
	var buf [8]byte
	off, err := r.Read(buf[:])
	if off != 8 || err != nil {
		return nil, fmt.Errorf("failed to read header: read %v, err %v", off, err)
	}
	if bytes.Compare(buf[:], []byte("gotrace\x00")) != 0 {
		return nil, fmt.Errorf("not a trace file")
	}

	// Read events.
	var events []RawEvent
	for {
		// Read event type and number of arguments (1 byte).
		off0 := off
		n, err := r.Read(buf[:1])
		if err == io.EOF {
			break
		}
		if err != nil || n != 1 {
			return nil, fmt.Errorf("failed to read trace at offset 0x%x: n=%v err=%v", off0, n, err)
		}
		off += n
		typ := buf[0] << 2 >> 2
		narg := buf[0]>>6 + 1
		ev := RawEvent{typ: typ, off: off0}
		if narg <= 3 {
			for i := 0; i < int(narg); i++ {
				var v uint64
				v, off, err = readVal(r, off)
				if err != nil {
					return nil, err
				}
				ev.args = append(ev.args, v)
			}
		} else {
			// If narg == 4, the first value is length of the event in bytes.
			var v uint64
			v, off, err = readVal(r, off)
			if err != nil {
				return nil, err
			}
			evLen := v
			off1 := off
			for evLen > uint64(off-off1) {
				v, off, err = readVal(r, off)
				if err != nil {
					return nil, err
				}
				ev.args = append(ev.args, v)
			}
			if evLen != uint64(off-off1) {
				return nil, fmt.Errorf("event has wrong length at offset 0x%x: want %v, got %v", off0, evLen, off-off1)
			}
		}
		events = append(events, ev)
	}
	return events, nil
}

// Parse events transforms raw events into events.
// It does analyze and verify per-event-type arguments.
func parseEvents(rawEvents []RawEvent) (events []*Event, err error) {
	var ticksPerSec, lastTs int64
	var lastG, timerGoid uint64
	var lastP int
	lastGs := make(map[int]uint64) // last goroutine running on P
	stacks := make(map[uint64][]*Frame)
	for _, raw := range rawEvents {
		if raw.typ == traceEvNone || raw.typ >= traceEvCount {
			err = fmt.Errorf("unknown event type %v at offset 0x%x", raw.typ, raw.off)
			return
		}
		desc := evDescriptions[raw.typ]
		if desc.name == "" {
			err = fmt.Errorf("missing description for event type %v", raw.typ)
			return
		}
		if raw.typ != traceEvStack {
			narg := len(desc.args)
			if desc.stack {
				narg++
			}
			if raw.typ != traceEvBatch && raw.typ != traceEvFrequency && raw.typ != traceEvTimerGoroutine {
				narg++ // timestamp
			}
			if len(raw.args) != narg {
				err = fmt.Errorf("%v has wrong number of arguments at offset 0x%x: want %v, got %v",
					desc.name, raw.off, narg, len(raw.args))
				return
			}
		}
		switch raw.typ {
		case traceEvBatch:
			lastGs[lastP] = lastG
			lastP = int(raw.args[0])
			lastG = lastGs[lastP]
			lastTs = int64(raw.args[1])
		case traceEvFrequency:
			ticksPerSec = int64(raw.args[0])
			if ticksPerSec <= 0 {
				err = fmt.Errorf("traceEvFrequency contains invalid frequency %v at offset 0x%x",
					ticksPerSec, raw.off)
				return
			}
		case traceEvTimerGoroutine:
			timerGoid = raw.args[0]
		case traceEvStack:
			if len(raw.args) < 2 {
				err = fmt.Errorf("traceEvStack has wrong number of arguments at offset 0x%x: want at least 2, got %v",
					raw.off, len(raw.args))
				return
			}
			size := raw.args[1]
			if size > 1000 {
				err = fmt.Errorf("traceEvStack has bad number of frames at offset 0x%x: %v",
					raw.off, size)
				return
			}
			id := raw.args[0]
			if id != 0 && size > 0 {
				stk := make([]*Frame, size)
				for i := 0; i < int(size); i++ {
					stk[i] = &Frame{pc: raw.args[i+2]}
				}
				stacks[id] = stk
			}
		default:
			e := &Event{off: raw.off, typ: raw.typ, p: lastP, g: lastG}
			e.ts = lastTs + int64(raw.args[0])
			lastTs = e.ts
			for i := range desc.args {
				e.args[i] = raw.args[i+1]
			}
			if desc.stack {
				e.stkID = raw.args[len(desc.args)+1]
			}
			switch raw.typ {
			case traceEvGoStart:
				lastG = e.args[0]
				e.g = lastG
			case traceEvGCStart, traceEvGCDone, traceEvGCScanStart, traceEvGCScanDone:
				e.g = 0
			case traceEvGoEnd, traceEvGoStop, traceEvGoSched, traceEvGoPreempt,
				traceEvGoSleep, traceEvGoBlock, traceEvGoBlockSend, traceEvGoBlockRecv,
				traceEvGoBlockSelect, traceEvGoBlockSync, traceEvGoBlockCond, traceEvGoBlockNet,
				traceEvGoSysBlock:
				lastG = 0
			}
			events = append(events, e)
		}
	}

	// Attach stack traces.
	for _, ev := range events {
		if ev.stkID != 0 {
			ev.stk = stacks[ev.stkID]
		}
	}

	// Sort by time and translate cpu ticks to real time.
	sort.Sort(EventList(events))
	if ticksPerSec == 0 {
		err = fmt.Errorf("no traceEvFrequency event")
		return
	}
	minTs := events[0].ts
	for _, ev := range events {
		ev.ts = (ev.ts - minTs) * 1e9 / ticksPerSec
		// Move timers and syscalls to separate fake Ps.
		if timerGoid != 0 && ev.g == timerGoid && ev.typ == traceEvGoUnblock {
			ev.p = timerP
		}
		if ev.typ == traceEvGoSysExit {
			ev.p = syscallP
			ev.g = ev.args[0]
		}
	}

	return
}

// postProcessTrace does inter-event verification and information restoration.
// The resulting trace is guaranteed to be consistent
// (for example, a P does not run two Gs at the same time, or a G is indeed
// blocked before an unblock event).
func postProcessTrace(events []*Event) error {
	const (
		gDead = iota
		gRunnable
		gRunning
		gWaiting
	)
	type gdesc struct {
		state   int
		ev      *Event
		evStart *Event
	}
	type pdesc struct {
		running bool
		g       uint64
		evGC    *Event
		evScan  *Event
		evSweep *Event
	}

	gs := make(map[uint64]gdesc)
	ps := make(map[int]pdesc)
	gs[0] = gdesc{state: gRunning}

	checkRunning := func(p pdesc, g gdesc, ev *Event) error {
		name := evDescriptions[ev.typ].name
		if g.state != gRunning {
			return fmt.Errorf("g %v is not running while %v (offset %v, time %v)", ev.g, name, ev.off, ev.ts)
		}
		if p.g != ev.g {
			return fmt.Errorf("p %v is not running g %v while %v (offset %v, time %v)", ev.p, ev.g, name, ev.off, ev.ts)
		}
		return nil
	}

	for _, ev := range events {
		g := gs[ev.g]
		p := ps[ev.p]

		switch ev.typ {
		case traceEvProcStart:
			if p.running {
				return fmt.Errorf("p %v is running before start (offset %v, time %v)", ev.p, ev.off, ev.ts)
			}
			p.running = true
		case traceEvProcStop:
			if !p.running {
				return fmt.Errorf("p %v is not running before stop (offset %v, time %v)", ev.p, ev.off, ev.ts)
			}
			if p.g != 0 {
				return fmt.Errorf("p %v is running a goroutine %v during stop (offset %v, time %v)", ev.p, p.g, ev.off, ev.ts)
			}
			p.running = false
		case traceEvGCStart:
			if p.evGC != nil {
				return fmt.Errorf("previous GC is not ended before a new one (offset %v, time %v)", ev.off, ev.ts)
			}
			p.evGC = ev
		case traceEvGCDone:
			if p.evGC == nil {
				return fmt.Errorf("bogus GC end (offset %v, time %v)", ev.off, ev.ts)
			}
			p.evGC.link = ev
			p.evGC = nil
		case traceEvGCScanStart:
			if p.evScan != nil {
				return fmt.Errorf("previous scanning is not ended before a new one (offset %v, time %v)", ev.off, ev.ts)
			}
			p.evScan = ev
		case traceEvGCScanDone:
			if p.evScan == nil {
				return fmt.Errorf("bogus scanning end (offset %v, time %v)", ev.off, ev.ts)
			}
			p.evScan.link = ev
			p.evScan = nil
		case traceEvGCSweepStart:
			if p.evSweep != nil {
				return fmt.Errorf("previous sweeping is not ended before a new one (offset %v, time %v)", ev.off, ev.ts)
			}
			p.evSweep = ev
		case traceEvGCSweepDone:
			if p.evSweep == nil {
				return fmt.Errorf("bogus sweeping end (offset %v, time %v)", ev.off, ev.ts)
			}
			p.evSweep.link = ev
			p.evSweep = nil
		case traceEvGoWaiting:
			g1 := gs[ev.args[0]]
			if g1.state != gRunnable {
				return fmt.Errorf("g %v is not runnable before traceEvGoWaiting (offset %v, time %v)", ev.args[0], ev.off, ev.ts)
			}
			g1.state = gWaiting
			gs[ev.args[0]] = g1
		case traceEvGoInSyscall:
			// this case is intentionally left blank
		case traceEvGoCreate:
			if err := checkRunning(p, g, ev); err != nil {
				return err
			}
			if _, ok := gs[ev.args[0]]; ok {
				return fmt.Errorf("g %v already exists (offset %v, time %v)", ev.args[0], ev.off, ev.ts)
			}
			gs[ev.args[0]] = gdesc{state: gRunnable, ev: ev}
		case traceEvGoStart:
			if g.state != gRunnable {
				return fmt.Errorf("g %v is not runnable before start (offset %v, time %v)", ev.g, ev.off, ev.ts)
			}
			if p.g != 0 {
				return fmt.Errorf("p %v is already running g %v while start g %v (offset %v, time %v)", ev.p, p.g, ev.g, ev.off, ev.ts)
			}
			g.state = gRunning
			g.evStart = ev
			p.g = ev.g
			if g.ev != nil {
				if g.ev.typ == traceEvGoCreate {
					// +1 because symblizer expects return pc.
					ev.stk = []*Frame{&Frame{pc: g.ev.args[1] + 1}}
				}
				g.ev.link = ev
				g.ev = nil
			}
		case traceEvGoEnd, traceEvGoStop:
			if err := checkRunning(p, g, ev); err != nil {
				return err
			}
			g.evStart.link = ev
			g.evStart = nil
			g.state = gDead
			p.g = 0
		case traceEvGoSched, traceEvGoPreempt:
			if err := checkRunning(p, g, ev); err != nil {
				return err
			}
			g.state = gRunnable
			g.evStart.link = ev
			g.evStart = nil
			p.g = 0
			g.ev = ev
		case traceEvGoUnblock:
			if g.state != gRunning {
				return fmt.Errorf("g %v is not running while unpark (offset %v, time %v)", ev.g, ev.off, ev.ts)
			}
			if ev.p != timerP && p.g != ev.g {
				return fmt.Errorf("p %v is not running g %v while unpark (offset %v, time %v)", ev.p, ev.g, ev.off, ev.ts)
			}
			g1 := gs[ev.args[0]]
			if g1.state != gWaiting {
				return fmt.Errorf("g %v is not waiting before unpark (offset %v, time %v)", ev.args[0], ev.off, ev.ts)
			}
			if g1.ev != nil && g1.ev.typ == traceEvGoBlockNet && ev.p != timerP {
				ev.p = netpollP
			}
			if g1.ev != nil {
				g1.ev.link = ev
			}
			g1.state = gRunnable
			g1.ev = ev
			gs[ev.args[0]] = g1
		case traceEvGoSysCall:
			if err := checkRunning(p, g, ev); err != nil {
				return err
			}
			g.ev = ev
		case traceEvGoSysBlock:
			if err := checkRunning(p, g, ev); err != nil {
				return err
			}
			g.state = gRunnable
			g.evStart.link = ev
			g.evStart = nil
			p.g = 0
		case traceEvGoSysExit:
			if g.state != gRunnable {
				return fmt.Errorf("g %v is not runnable during syscall exit (offset %v, time %v)", ev.g, ev.off, ev.ts)
			}
			if g.ev != nil && g.ev.typ == traceEvGoSysCall {
				g.ev.link = ev
			}
			g.ev = ev
		case traceEvGoSleep, traceEvGoBlock, traceEvGoBlockSend, traceEvGoBlockRecv,
			traceEvGoBlockSelect, traceEvGoBlockSync, traceEvGoBlockCond, traceEvGoBlockNet:
			if err := checkRunning(p, g, ev); err != nil {
				return err
			}
			g.state = gWaiting
			g.ev = ev
			g.evStart.link = ev
			g.evStart = nil
			p.g = 0
		}

		gs[ev.g] = g
		ps[ev.p] = p
	}

	return nil
}

// symbolizeTrace attaches func/file/line info to stack traces.
func symbolizeTrace(events []*Event, bin string) error {
	// First, collect and dedup all pcs.
	pcs := make(map[uint64]*Frame)
	for _, ev := range events {
		for _, f := range ev.stk {
			pcs[f.pc] = nil
		}
	}

	// Start addr2line.
	cmd := exec.Command("go", "tool", "addr2line", bin)
	in, err := cmd.StdinPipe()
	if err != nil {
		return fmt.Errorf("failed to pipe addr2line stdin: %v", err)
	}
	out, err := cmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("failed to pipe addr2line stdout: %v", err)
	}
	err = cmd.Start()
	if err != nil {
		return fmt.Errorf("failed to start addr2line: %v", err)
	}
	outb := bufio.NewReader(out)

	// Write all pcs to addr2line.
	// Need to copy pcs to an array, because map iteration order is non-deterministic.
	var pcArray []uint64
	for pc := range pcs {
		pcArray = append(pcArray, pc)
		_, err := fmt.Fprintf(in, "0x%x\n", pc-1)
		if err != nil {
			return fmt.Errorf("failed to write to addr2line: %v", err)
		}
	}
	in.Close()

	// Read in answers.
	for _, pc := range pcArray {
		fn, err := outb.ReadString('\n')
		if err != nil {
			return fmt.Errorf("failed to read from addr2line: %v", err)
		}
		file, err := outb.ReadString('\n')
		if err != nil {
			return fmt.Errorf("failed to read from addr2line: %v", err)
		}
		f := &Frame{pc: pc}
		f.fn = fn[:len(fn)-1]
		f.file = file[:len(file)-1]
		if colon := strings.LastIndex(f.file, ":"); colon != -1 {
			ln, err := strconv.Atoi(f.file[colon+1:])
			if err == nil {
				f.file = f.file[:colon]
				f.line = ln
			}
		}
		pcs[pc] = f
	}
	cmd.Wait()

	// Replace frames in events array.
	for _, ev := range events {
		for i, f := range ev.stk {
			ev.stk[i] = pcs[f.pc]
		}
	}

	return nil
}

// readVal reads unsigned base-128 value from r.
func readVal(r io.Reader, off0 int) (v uint64, off int, err error) {
	off = off0
	for i := 0; i < 10; i++ {
		var buf [1]byte
		var n int
		n, err = r.Read(buf[:])
		if err != nil || n != 1 {
			return 0, 0, fmt.Errorf("failed to read trace at offset: read %v, error %v", off0, n, err)
		}
		off++
		v |= uint64(buf[0]&0x7f) << (uint(i) * 7)
		if buf[0]&0x80 == 0 {
			return
		}
	}
	return 0, 0, fmt.Errorf("bad value at offset 0x%x", off0)
}

type EventList []*Event

func (l EventList) Len() int {
	return len(l)
}

func (l EventList) Less(i, j int) bool {
	return l[i].ts < l[j].ts
}

func (l EventList) Swap(i, j int) {
	l[i], l[j] = l[j], l[i]
}

// Event types in the trace.
// Verbatim copy from src/runtime/trace.go.
const (
	traceEvNone           = 0  // unused
	traceEvBatch          = 1  // start of per-P batch of events [pid, timestamp]
	traceEvFrequency      = 2  // contains tracer timer frequency [frequency (ticks per second)]
	traceEvStack          = 3  // stack [stack id, number of PCs, array of PCs]
	traceEvGomaxprocs     = 4  // current value of GOMAXPROCS [timestamp, GOMAXPROCS, stack id]
	traceEvProcStart      = 5  // start of P [timestamp]
	traceEvProcStop       = 6  // stop of P [timestamp]
	traceEvGCStart        = 7  // GC start [timestamp, stack id]
	traceEvGCDone         = 8  // GC done [timestamp]
	traceEvGCScanStart    = 9  // GC scan start [timestamp]
	traceEvGCScanDone     = 10 // GC scan done [timestamp]
	traceEvGCSweepStart   = 11 // GC sweep start [timestamp, stack id]
	traceEvGCSweepDone    = 12 // GC sweep done [timestamp]
	traceEvGoCreate       = 13 // goroutine creation [timestamp, new goroutine id, start PC, stack id]
	traceEvGoStart        = 14 // goroutine starts running [timestamp, goroutine id]
	traceEvGoEnd          = 15 // goroutine ends [timestamp]
	traceEvGoStop         = 16 // goroutine stops (like in select{}) [timestamp, stack]
	traceEvGoSched        = 17 // goroutine calls Gosched [timestamp, stack]
	traceEvGoPreempt      = 18 // goroutine is preempted [timestamp, stack]
	traceEvGoSleep        = 19 // goroutine calls Sleep [timestamp, stack]
	traceEvGoBlock        = 20 // goroutine blocks [timestamp, stack]
	traceEvGoUnblock      = 21 // goroutine is unblocked [timestamp, goroutine id, stack]
	traceEvGoBlockSend    = 22 // goroutine blocks on chan send [timestamp, stack]
	traceEvGoBlockRecv    = 23 // goroutine blocks on chan recv [timestamp, stack]
	traceEvGoBlockSelect  = 24 // goroutine blocks on select [timestamp, stack]
	traceEvGoBlockSync    = 25 // goroutine blocks on Mutex/RWMutex [timestamp, stack]
	traceEvGoBlockCond    = 26 // goroutine blocks on Cond [timestamp, stack]
	traceEvGoBlockNet     = 27 // goroutine blocks on network [timestamp, stack]
	traceEvGoSysCall      = 28 // syscall enter [timestamp, stack]
	traceEvGoSysExit      = 29 // syscall exit [timestamp, goroutine id]
	traceEvGoSysBlock     = 30 // syscall blocks [timestamp, stack]
	traceEvGoWaiting      = 31 // denotes that goroutine is blocked when tracing starts [goroutine id]
	traceEvGoInSyscall    = 32 // denotes that goroutine is in syscall when tracing starts [goroutine id]
	traceEvHeapAlloc      = 33 // memstats.heap_alloc change [timestamp, heap_alloc]
	traceEvNextGC         = 34 // memstats.next_gc change [timestamp, next_gc]
	traceEvTimerGoroutine = 35 // denotes timer goroutine [timer goroutine id]
	traceEvCount          = 36
)

var evDescriptions = [traceEvCount]struct {
	name  string
	stack bool
	args  []string
}{
	traceEvNone:           {"None", false, []string{}},
	traceEvBatch:          {"Batch", false, []string{"p", "ticks"}},
	traceEvFrequency:      {"Frequency", false, []string{"freq"}},
	traceEvStack:          {"Stack", false, []string{"id", "siz"}},
	traceEvGomaxprocs:     {"Gomaxprocs", true, []string{"procs"}},
	traceEvProcStart:      {"ProcStart", false, []string{}},
	traceEvProcStop:       {"ProcStop", false, []string{}},
	traceEvGCStart:        {"GCStart", true, []string{}},
	traceEvGCDone:         {"GCDone", false, []string{}},
	traceEvGCScanStart:    {"GCScanStart", false, []string{}},
	traceEvGCScanDone:     {"GCScanDone", false, []string{}},
	traceEvGCSweepStart:   {"GCSweepStart", true, []string{}},
	traceEvGCSweepDone:    {"GCSweepDone", false, []string{}},
	traceEvGoCreate:       {"GoCreate", true, []string{"g", "pc"}},
	traceEvGoStart:        {"GoStart", false, []string{"g"}},
	traceEvGoEnd:          {"GoEnd", false, []string{}},
	traceEvGoStop:         {"GoStop", true, []string{}},
	traceEvGoSched:        {"GoSched", true, []string{}},
	traceEvGoPreempt:      {"GoPreempt", true, []string{}},
	traceEvGoSleep:        {"GoSleep", true, []string{}},
	traceEvGoBlock:        {"GoBlock", true, []string{}},
	traceEvGoUnblock:      {"GoUnblock", true, []string{"g"}},
	traceEvGoBlockSend:    {"GoBlockSend", true, []string{}},
	traceEvGoBlockRecv:    {"GoBlockRecv", true, []string{}},
	traceEvGoBlockSelect:  {"GoBlockSelect", true, []string{}},
	traceEvGoBlockSync:    {"GoBlockSync", true, []string{}},
	traceEvGoBlockCond:    {"GoBlockCond", true, []string{}},
	traceEvGoBlockNet:     {"GoBlockNet", true, []string{}},
	traceEvGoSysCall:      {"GoSysCall", true, []string{}},
	traceEvGoSysExit:      {"GoSysExit", false, []string{"g"}},
	traceEvGoSysBlock:     {"GoSysBlock", true, []string{}},
	traceEvGoWaiting:      {"GoWaiting", false, []string{"g"}},
	traceEvGoInSyscall:    {"GoInSyscall", false, []string{"g"}},
	traceEvHeapAlloc:      {"HeapAlloc", false, []string{"mem"}},
	traceEvNextGC:         {"NextGC", false, []string{"mem"}},
	traceEvTimerGoroutine: {"TimerGoroutine", false, []string{"g"}},
}

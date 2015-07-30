// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trace

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"os"
	"os/exec"
	"sort"
	"strconv"
	"strings"
)

// Event describes one event in the trace.
type Event struct {
	Off   int       // offset in input file (for debugging and error reporting)
	Type  byte      // one of Ev*
	Seq   int64     // sequence number
	Ts    int64     // timestamp in nanoseconds
	P     int       // P on which the event happened (can be one of TimerP, NetpollP, SyscallP)
	G     uint64    // G on which the event happened
	StkID uint64    // unique stack ID
	Stk   []*Frame  // stack trace (can be empty)
	Args  [3]uint64 // event-type-specific arguments
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
	Link *Event
}

// Frame is a frame in stack traces.
type Frame struct {
	PC   uint64
	Fn   string
	File string
	Line int
}

const (
	// Special P identifiers:
	FakeP    = 1000000 + iota
	TimerP   // depicts timer unblocks
	NetpollP // depicts network unblocks
	SyscallP // depicts returns from syscalls
)

// Parse parses, post-processes and verifies the trace.
func Parse(r io.Reader) ([]*Event, error) {
	rawEvents, err := readTrace(r)
	if err != nil {
		return nil, err
	}
	events, err := parseEvents(rawEvents)
	if err != nil {
		return nil, err
	}
	events, err = removeFutile(events)
	if err != nil {
		return nil, err
	}
	err = postProcessTrace(events)
	if err != nil {
		return nil, err
	}
	return events, nil
}

// rawEvent is a helper type used during parsing.
type rawEvent struct {
	off  int
	typ  byte
	args []uint64
}

// readTrace does wire-format parsing and verification.
// It does not care about specific event types and argument meaning.
func readTrace(r io.Reader) ([]rawEvent, error) {
	// Read and validate trace header.
	var buf [16]byte
	off, err := r.Read(buf[:])
	if off != 16 || err != nil {
		return nil, fmt.Errorf("failed to read header: read %v, err %v", off, err)
	}
	if bytes.Compare(buf[:], []byte("go 1.5 trace\x00\x00\x00\x00")) != 0 {
		return nil, fmt.Errorf("not a trace file")
	}

	// Read events.
	var events []rawEvent
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
		narg := buf[0] >> 6
		ev := rawEvent{typ: typ, off: off0}
		if narg < 3 {
			for i := 0; i < int(narg)+2; i++ { // sequence number and time stamp are present but not counted in narg
				var v uint64
				v, off, err = readVal(r, off)
				if err != nil {
					return nil, err
				}
				ev.args = append(ev.args, v)
			}
		} else {
			// If narg == 3, the first value is length of the event in bytes.
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
func parseEvents(rawEvents []rawEvent) (events []*Event, err error) {
	var ticksPerSec, lastSeq, lastTs int64
	var lastG, timerGoid uint64
	var lastP int
	lastGs := make(map[int]uint64) // last goroutine running on P
	stacks := make(map[uint64][]*Frame)
	for _, raw := range rawEvents {
		if raw.typ == EvNone || raw.typ >= EvCount {
			err = fmt.Errorf("unknown event type %v at offset 0x%x", raw.typ, raw.off)
			return
		}
		desc := EventDescriptions[raw.typ]
		if desc.Name == "" {
			err = fmt.Errorf("missing description for event type %v", raw.typ)
			return
		}
		if raw.typ != EvStack {
			narg := len(desc.Args)
			if desc.Stack {
				narg++
			}
			if raw.typ != EvBatch && raw.typ != EvFrequency && raw.typ != EvTimerGoroutine {
				narg++ // sequence number
				narg++ // timestamp
			}
			if len(raw.args) != narg {
				err = fmt.Errorf("%v has wrong number of arguments at offset 0x%x: want %v, got %v",
					desc.Name, raw.off, narg, len(raw.args))
				return
			}
		}
		switch raw.typ {
		case EvBatch:
			lastGs[lastP] = lastG
			lastP = int(raw.args[0])
			lastG = lastGs[lastP]
			lastSeq = int64(raw.args[1])
			lastTs = int64(raw.args[2])
		case EvFrequency:
			ticksPerSec = int64(raw.args[0])
			if ticksPerSec <= 0 {
				// The most likely cause for this is tick skew on different CPUs.
				// For example, solaris/amd64 seems to have wildly different
				// ticks on different CPUs.
				err = ErrTimeOrder
				return
			}
		case EvTimerGoroutine:
			timerGoid = raw.args[0]
		case EvStack:
			if len(raw.args) < 2 {
				err = fmt.Errorf("EvStack has wrong number of arguments at offset 0x%x: want at least 2, got %v",
					raw.off, len(raw.args))
				return
			}
			size := raw.args[1]
			if size > 1000 {
				err = fmt.Errorf("EvStack has bad number of frames at offset 0x%x: %v",
					raw.off, size)
				return
			}
			if uint64(len(raw.args)) != size+2 {
				err = fmt.Errorf("EvStack has wrong number of arguments at offset 0x%x: want %v, got %v",
					raw.off, size+2, len(raw.args))
				return
			}
			id := raw.args[0]
			if id != 0 && size > 0 {
				stk := make([]*Frame, size)
				for i := 0; i < int(size); i++ {
					stk[i] = &Frame{PC: raw.args[i+2]}
				}
				stacks[id] = stk
			}
		default:
			e := &Event{Off: raw.off, Type: raw.typ, P: lastP, G: lastG}
			e.Seq = lastSeq + int64(raw.args[0])
			e.Ts = lastTs + int64(raw.args[1])
			lastSeq = e.Seq
			lastTs = e.Ts
			for i := range desc.Args {
				e.Args[i] = raw.args[i+2]
			}
			if desc.Stack {
				e.StkID = raw.args[len(desc.Args)+2]
			}
			switch raw.typ {
			case EvGoStart:
				lastG = e.Args[0]
				e.G = lastG
			case EvGCStart, EvGCDone, EvGCScanStart, EvGCScanDone:
				e.G = 0
			case EvGoEnd, EvGoStop, EvGoSched, EvGoPreempt,
				EvGoSleep, EvGoBlock, EvGoBlockSend, EvGoBlockRecv,
				EvGoBlockSelect, EvGoBlockSync, EvGoBlockCond, EvGoBlockNet,
				EvGoSysBlock:
				lastG = 0
			case EvGoSysExit:
				// EvGoSysExit emission is delayed until the thread has a P.
				// Give it the real sequence number and time stamp.
				e.Seq = int64(e.Args[1])
				if e.Args[2] != 0 {
					e.Ts = int64(e.Args[2])
				}
			}
			events = append(events, e)
		}
	}
	if len(events) == 0 {
		err = fmt.Errorf("trace is empty")
		return
	}

	// Attach stack traces.
	for _, ev := range events {
		if ev.StkID != 0 {
			ev.Stk = stacks[ev.StkID]
		}
	}

	// Sort by sequence number and translate cpu ticks to real time.
	sort.Sort(eventList(events))
	if ticksPerSec == 0 {
		err = fmt.Errorf("no EvFrequency event")
		return
	}
	minTs := events[0].Ts
	for _, ev := range events {
		ev.Ts = (ev.Ts - minTs) * 1e9 / ticksPerSec
		// Move timers and syscalls to separate fake Ps.
		if timerGoid != 0 && ev.G == timerGoid && ev.Type == EvGoUnblock {
			ev.P = TimerP
		}
		if ev.Type == EvGoSysExit {
			ev.P = SyscallP
			ev.G = ev.Args[0]
		}
	}

	return
}

// removeFutile removes all constituents of futile wakeups (block, unblock, start).
// For example, a goroutine was unblocked on a mutex, but another goroutine got
// ahead and acquired the mutex before the first goroutine is scheduled,
// so the first goroutine has to block again. Such wakeups happen on buffered
// channels and sync.Mutex, but are generally not interesting for end user.
func removeFutile(events []*Event) ([]*Event, error) {
	// Two non-trivial aspects:
	// 1. A goroutine can be preempted during a futile wakeup and migrate to another P.
	//	We want to remove all of that.
	// 2. Tracing can start in the middle of a futile wakeup.
	//	That is, we can see a futile wakeup event w/o the actual wakeup before it.
	// postProcessTrace runs after us and ensures that we leave the trace in a consistent state.

	// Phase 1: determine futile wakeup sequences.
	type G struct {
		futile bool
		wakeup []*Event // wakeup sequence (subject for removal)
	}
	gs := make(map[uint64]G)
	futile := make(map[*Event]bool)
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
		case EvGoBlock, EvGoBlockSend, EvGoBlockRecv, EvGoBlockSelect, EvGoBlockSync, EvGoBlockCond:
			g := gs[ev.G]
			if g.futile {
				futile[ev] = true
				for _, ev1 := range g.wakeup {
					futile[ev1] = true
				}
			}
			delete(gs, ev.G)
		}
	}

	// Phase 2: remove futile wakeup sequences.
	newEvents := events[:0] // overwrite the original slice
	for _, ev := range events {
		if !futile[ev] {
			newEvents = append(newEvents, ev)
		}
	}
	return newEvents, nil
}

// ErrTimeOrder is returned by Parse when the trace contains
// time stamps that do not respect actual event ordering.
var ErrTimeOrder = fmt.Errorf("time stamps out of order")

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
		state    int
		ev       *Event
		evStart  *Event
		evCreate *Event
	}
	type pdesc struct {
		running bool
		g       uint64
		evScan  *Event
		evSweep *Event
	}

	gs := make(map[uint64]gdesc)
	ps := make(map[int]pdesc)
	gs[0] = gdesc{state: gRunning}
	var evGC *Event

	checkRunning := func(p pdesc, g gdesc, ev *Event, allowG0 bool) error {
		name := EventDescriptions[ev.Type].Name
		if g.state != gRunning {
			return fmt.Errorf("g %v is not running while %v (offset %v, time %v)", ev.G, name, ev.Off, ev.Ts)
		}
		if p.g != ev.G {
			return fmt.Errorf("p %v is not running g %v while %v (offset %v, time %v)", ev.P, ev.G, name, ev.Off, ev.Ts)
		}
		if !allowG0 && ev.G == 0 {
			return fmt.Errorf("g 0 did %v (offset %v, time %v)", EventDescriptions[ev.Type].Name, ev.Off, ev.Ts)
		}
		return nil
	}

	for _, ev := range events {
		g := gs[ev.G]
		p := ps[ev.P]

		switch ev.Type {
		case EvProcStart:
			if p.running {
				return fmt.Errorf("p %v is running before start (offset %v, time %v)", ev.P, ev.Off, ev.Ts)
			}
			p.running = true
		case EvProcStop:
			if !p.running {
				return fmt.Errorf("p %v is not running before stop (offset %v, time %v)", ev.P, ev.Off, ev.Ts)
			}
			if p.g != 0 {
				return fmt.Errorf("p %v is running a goroutine %v during stop (offset %v, time %v)", ev.P, p.g, ev.Off, ev.Ts)
			}
			p.running = false
		case EvGCStart:
			if evGC != nil {
				return fmt.Errorf("previous GC is not ended before a new one (offset %v, time %v)", ev.Off, ev.Ts)
			}
			evGC = ev
		case EvGCDone:
			if evGC == nil {
				return fmt.Errorf("bogus GC end (offset %v, time %v)", ev.Off, ev.Ts)
			}
			evGC.Link = ev
			evGC = nil
		case EvGCScanStart:
			if p.evScan != nil {
				return fmt.Errorf("previous scanning is not ended before a new one (offset %v, time %v)", ev.Off, ev.Ts)
			}
			p.evScan = ev
		case EvGCScanDone:
			if p.evScan == nil {
				return fmt.Errorf("bogus scanning end (offset %v, time %v)", ev.Off, ev.Ts)
			}
			p.evScan.Link = ev
			p.evScan = nil
		case EvGCSweepStart:
			if p.evSweep != nil {
				return fmt.Errorf("previous sweeping is not ended before a new one (offset %v, time %v)", ev.Off, ev.Ts)
			}
			p.evSweep = ev
		case EvGCSweepDone:
			if p.evSweep == nil {
				return fmt.Errorf("bogus sweeping end (offset %v, time %v)", ev.Off, ev.Ts)
			}
			p.evSweep.Link = ev
			p.evSweep = nil
		case EvGoWaiting:
			g1 := gs[ev.Args[0]]
			if g1.state != gRunnable {
				return fmt.Errorf("g %v is not runnable before EvGoWaiting (offset %v, time %v)", ev.Args[0], ev.Off, ev.Ts)
			}
			g1.state = gWaiting
			gs[ev.Args[0]] = g1
		case EvGoInSyscall:
			g1 := gs[ev.Args[0]]
			if g1.state != gRunnable {
				return fmt.Errorf("g %v is not runnable before EvGoInSyscall (offset %v, time %v)", ev.Args[0], ev.Off, ev.Ts)
			}
			g1.state = gWaiting
			gs[ev.Args[0]] = g1
		case EvGoCreate:
			if err := checkRunning(p, g, ev, true); err != nil {
				return err
			}
			if _, ok := gs[ev.Args[0]]; ok {
				return fmt.Errorf("g %v already exists (offset %v, time %v)", ev.Args[0], ev.Off, ev.Ts)
			}
			gs[ev.Args[0]] = gdesc{state: gRunnable, ev: ev, evCreate: ev}
		case EvGoStart:
			if g.state != gRunnable {
				return fmt.Errorf("g %v is not runnable before start (offset %v, time %v)", ev.G, ev.Off, ev.Ts)
			}
			if p.g != 0 {
				return fmt.Errorf("p %v is already running g %v while start g %v (offset %v, time %v)", ev.P, p.g, ev.G, ev.Off, ev.Ts)
			}
			g.state = gRunning
			g.evStart = ev
			p.g = ev.G
			if g.evCreate != nil {
				// +1 because symbolizer expects return pc.
				ev.Stk = []*Frame{&Frame{PC: g.evCreate.Args[1] + 1}}
				g.evCreate = nil
			}

			if g.ev != nil {
				g.ev.Link = ev
				g.ev = nil
			}
		case EvGoEnd, EvGoStop:
			if err := checkRunning(p, g, ev, false); err != nil {
				return err
			}
			g.evStart.Link = ev
			g.evStart = nil
			g.state = gDead
			p.g = 0
		case EvGoSched, EvGoPreempt:
			if err := checkRunning(p, g, ev, false); err != nil {
				return err
			}
			g.state = gRunnable
			g.evStart.Link = ev
			g.evStart = nil
			p.g = 0
			g.ev = ev
		case EvGoUnblock:
			if g.state != gRunning {
				return fmt.Errorf("g %v is not running while unpark (offset %v, time %v)", ev.G, ev.Off, ev.Ts)
			}
			if ev.P != TimerP && p.g != ev.G {
				return fmt.Errorf("p %v is not running g %v while unpark (offset %v, time %v)", ev.P, ev.G, ev.Off, ev.Ts)
			}
			g1 := gs[ev.Args[0]]
			if g1.state != gWaiting {
				return fmt.Errorf("g %v is not waiting before unpark (offset %v, time %v)", ev.Args[0], ev.Off, ev.Ts)
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
			if err := checkRunning(p, g, ev, false); err != nil {
				return err
			}
			g.ev = ev
		case EvGoSysBlock:
			if err := checkRunning(p, g, ev, false); err != nil {
				return err
			}
			g.state = gWaiting
			g.evStart.Link = ev
			g.evStart = nil
			p.g = 0
		case EvGoSysExit:
			if g.state != gWaiting {
				return fmt.Errorf("g %v is not waiting during syscall exit (offset %v, time %v)", ev.G, ev.Off, ev.Ts)
			}
			if g.ev != nil && g.ev.Type == EvGoSysCall {
				g.ev.Link = ev
			}
			g.state = gRunnable
			g.ev = ev
		case EvGoSleep, EvGoBlock, EvGoBlockSend, EvGoBlockRecv,
			EvGoBlockSelect, EvGoBlockSync, EvGoBlockCond, EvGoBlockNet:
			if err := checkRunning(p, g, ev, false); err != nil {
				return err
			}
			g.state = gWaiting
			g.ev = ev
			g.evStart.Link = ev
			g.evStart = nil
			p.g = 0
		}

		gs[ev.G] = g
		ps[ev.P] = p
	}

	// TODO(dvyukov): restore stacks for EvGoStart events.
	// TODO(dvyukov): test that all EvGoStart events has non-nil Link.

	// Last, after all the other consistency checks,
	// make sure time stamps respect sequence numbers.
	// The tests will skip (not fail) the test case if they see this error,
	// so check everything else that could possibly be wrong first.
	lastTs := int64(0)
	for _, ev := range events {
		if ev.Ts < lastTs {
			return ErrTimeOrder
		}
		lastTs = ev.Ts
	}

	return nil
}

// symbolizeTrace attaches func/file/line info to stack traces.
func Symbolize(events []*Event, bin string) error {
	// First, collect and dedup all pcs.
	pcs := make(map[uint64]*Frame)
	for _, ev := range events {
		for _, f := range ev.Stk {
			pcs[f.PC] = nil
		}
	}

	// Start addr2line.
	cmd := exec.Command("go", "tool", "addr2line", bin)
	in, err := cmd.StdinPipe()
	if err != nil {
		return fmt.Errorf("failed to pipe addr2line stdin: %v", err)
	}
	cmd.Stderr = os.Stderr
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
		f := &Frame{PC: pc}
		f.Fn = fn[:len(fn)-1]
		f.File = file[:len(file)-1]
		if colon := strings.LastIndex(f.File, ":"); colon != -1 {
			ln, err := strconv.Atoi(f.File[colon+1:])
			if err == nil {
				f.File = f.File[:colon]
				f.Line = ln
			}
		}
		pcs[pc] = f
	}
	cmd.Wait()

	// Replace frames in events array.
	for _, ev := range events {
		for i, f := range ev.Stk {
			ev.Stk[i] = pcs[f.PC]
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
			return 0, 0, fmt.Errorf("failed to read trace at offset %d: read %v, error %v", off0, n, err)
		}
		off++
		v |= uint64(buf[0]&0x7f) << (uint(i) * 7)
		if buf[0]&0x80 == 0 {
			return
		}
	}
	return 0, 0, fmt.Errorf("bad value at offset 0x%x", off0)
}

type eventList []*Event

func (l eventList) Len() int {
	return len(l)
}

func (l eventList) Less(i, j int) bool {
	return l[i].Seq < l[j].Seq
}

func (l eventList) Swap(i, j int) {
	l[i], l[j] = l[j], l[i]
}

// Print dumps events to stdout. For debugging.
func Print(events []*Event) {
	for _, ev := range events {
		desc := EventDescriptions[ev.Type]
		fmt.Printf("%v %v p=%v g=%v off=%v", ev.Ts, desc.Name, ev.P, ev.G, ev.Off)
		for i, a := range desc.Args {
			fmt.Printf(" %v=%v", a, ev.Args[i])
		}
		fmt.Printf("\n")
	}
}

// Event types in the trace.
// Verbatim copy from src/runtime/trace.go.
const (
	EvNone           = 0  // unused
	EvBatch          = 1  // start of per-P batch of events [pid, timestamp]
	EvFrequency      = 2  // contains tracer timer frequency [frequency (ticks per second)]
	EvStack          = 3  // stack [stack id, number of PCs, array of PCs]
	EvGomaxprocs     = 4  // current value of GOMAXPROCS [timestamp, GOMAXPROCS, stack id]
	EvProcStart      = 5  // start of P [timestamp, thread id]
	EvProcStop       = 6  // stop of P [timestamp]
	EvGCStart        = 7  // GC start [timestamp, stack id]
	EvGCDone         = 8  // GC done [timestamp]
	EvGCScanStart    = 9  // GC scan start [timestamp]
	EvGCScanDone     = 10 // GC scan done [timestamp]
	EvGCSweepStart   = 11 // GC sweep start [timestamp, stack id]
	EvGCSweepDone    = 12 // GC sweep done [timestamp]
	EvGoCreate       = 13 // goroutine creation [timestamp, new goroutine id, start PC, stack id]
	EvGoStart        = 14 // goroutine starts running [timestamp, goroutine id]
	EvGoEnd          = 15 // goroutine ends [timestamp]
	EvGoStop         = 16 // goroutine stops (like in select{}) [timestamp, stack]
	EvGoSched        = 17 // goroutine calls Gosched [timestamp, stack]
	EvGoPreempt      = 18 // goroutine is preempted [timestamp, stack]
	EvGoSleep        = 19 // goroutine calls Sleep [timestamp, stack]
	EvGoBlock        = 20 // goroutine blocks [timestamp, stack]
	EvGoUnblock      = 21 // goroutine is unblocked [timestamp, goroutine id, stack]
	EvGoBlockSend    = 22 // goroutine blocks on chan send [timestamp, stack]
	EvGoBlockRecv    = 23 // goroutine blocks on chan recv [timestamp, stack]
	EvGoBlockSelect  = 24 // goroutine blocks on select [timestamp, stack]
	EvGoBlockSync    = 25 // goroutine blocks on Mutex/RWMutex [timestamp, stack]
	EvGoBlockCond    = 26 // goroutine blocks on Cond [timestamp, stack]
	EvGoBlockNet     = 27 // goroutine blocks on network [timestamp, stack]
	EvGoSysCall      = 28 // syscall enter [timestamp, stack]
	EvGoSysExit      = 29 // syscall exit [timestamp, goroutine id, real timestamp]
	EvGoSysBlock     = 30 // syscall blocks [timestamp]
	EvGoWaiting      = 31 // denotes that goroutine is blocked when tracing starts [goroutine id]
	EvGoInSyscall    = 32 // denotes that goroutine is in syscall when tracing starts [goroutine id]
	EvHeapAlloc      = 33 // memstats.heap_alloc change [timestamp, heap_alloc]
	EvNextGC         = 34 // memstats.next_gc change [timestamp, next_gc]
	EvTimerGoroutine = 35 // denotes timer goroutine [timer goroutine id]
	EvFutileWakeup   = 36 // denotes that the previous wakeup of this goroutine was futile [timestamp]
	EvCount          = 37
)

var EventDescriptions = [EvCount]struct {
	Name  string
	Stack bool
	Args  []string
}{
	EvNone:           {"None", false, []string{}},
	EvBatch:          {"Batch", false, []string{"p", "seq", "ticks"}},
	EvFrequency:      {"Frequency", false, []string{"freq", "unused"}},
	EvStack:          {"Stack", false, []string{"id", "siz"}},
	EvGomaxprocs:     {"Gomaxprocs", true, []string{"procs"}},
	EvProcStart:      {"ProcStart", false, []string{"thread"}},
	EvProcStop:       {"ProcStop", false, []string{}},
	EvGCStart:        {"GCStart", true, []string{}},
	EvGCDone:         {"GCDone", false, []string{}},
	EvGCScanStart:    {"GCScanStart", false, []string{}},
	EvGCScanDone:     {"GCScanDone", false, []string{}},
	EvGCSweepStart:   {"GCSweepStart", true, []string{}},
	EvGCSweepDone:    {"GCSweepDone", false, []string{}},
	EvGoCreate:       {"GoCreate", true, []string{"g", "pc"}},
	EvGoStart:        {"GoStart", false, []string{"g"}},
	EvGoEnd:          {"GoEnd", false, []string{}},
	EvGoStop:         {"GoStop", true, []string{}},
	EvGoSched:        {"GoSched", true, []string{}},
	EvGoPreempt:      {"GoPreempt", true, []string{}},
	EvGoSleep:        {"GoSleep", true, []string{}},
	EvGoBlock:        {"GoBlock", true, []string{}},
	EvGoUnblock:      {"GoUnblock", true, []string{"g"}},
	EvGoBlockSend:    {"GoBlockSend", true, []string{}},
	EvGoBlockRecv:    {"GoBlockRecv", true, []string{}},
	EvGoBlockSelect:  {"GoBlockSelect", true, []string{}},
	EvGoBlockSync:    {"GoBlockSync", true, []string{}},
	EvGoBlockCond:    {"GoBlockCond", true, []string{}},
	EvGoBlockNet:     {"GoBlockNet", true, []string{}},
	EvGoSysCall:      {"GoSysCall", true, []string{}},
	EvGoSysExit:      {"GoSysExit", false, []string{"g", "seq", "ts"}},
	EvGoSysBlock:     {"GoSysBlock", false, []string{}},
	EvGoWaiting:      {"GoWaiting", false, []string{"g"}},
	EvGoInSyscall:    {"GoInSyscall", false, []string{"g"}},
	EvHeapAlloc:      {"HeapAlloc", false, []string{"mem"}},
	EvNextGC:         {"NextGC", false, []string{"mem"}},
	EvTimerGoroutine: {"TimerGoroutine", false, []string{"g", "unused"}},
	EvFutileWakeup:   {"FutileWakeup", false, []string{}},
}

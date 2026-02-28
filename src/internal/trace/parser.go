// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trace

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	_ "unsafe"
)

func goCmd() string {
	var exeSuffix string
	if runtime.GOOS == "windows" {
		exeSuffix = ".exe"
	}
	path := filepath.Join(runtime.GOROOT(), "bin", "go"+exeSuffix)
	if _, err := os.Stat(path); err == nil {
		return path
	}
	return "go"
}

// Event describes one event in the trace.
type Event struct {
	Off   int       // offset in input file (for debugging and error reporting)
	Type  byte      // one of Ev*
	seq   int64     // sequence number
	Ts    int64     // timestamp in nanoseconds
	P     int       // P on which the event happened (can be one of TimerP, NetpollP, SyscallP)
	G     uint64    // G on which the event happened
	StkID uint64    // unique stack ID
	Stk   []*Frame  // stack trace (can be empty)
	Args  [3]uint64 // event-type-specific arguments
	SArgs []string  // event-type-specific string args
	// linked event (can be nil), depends on event type:
	// for GCStart: the GCStop
	// for GCSTWStart: the GCSTWDone
	// for GCSweepStart: the GCSweepDone
	// for GoCreate: first GoStart of the created goroutine
	// for GoStart/GoStartLabel: the associated GoEnd, GoBlock or other blocking event
	// for GoSched/GoPreempt: the next GoStart
	// for GoBlock and other blocking events: the unblock event
	// for GoUnblock: the associated GoStart
	// for blocking GoSysCall: the associated GoSysExit
	// for GoSysExit: the next GoStart
	// for GCMarkAssistStart: the associated GCMarkAssistDone
	// for UserTaskCreate: the UserTaskEnd
	// for UserRegion: if the start region, the corresponding UserRegion end event
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
	GCP      // depicts GC state
)

// ParseResult is the result of Parse.
type ParseResult struct {
	// Events is the sorted list of Events in the trace.
	Events []*Event
	// Stacks is the stack traces keyed by stack IDs from the trace.
	Stacks map[uint64][]*Frame
}

// Parse parses, post-processes and verifies the trace.
func Parse(r io.Reader, bin string) (ParseResult, error) {
	ver, res, err := parse(r, bin)
	if err != nil {
		return ParseResult{}, err
	}
	if ver < 1007 && bin == "" {
		return ParseResult{}, fmt.Errorf("for traces produced by go 1.6 or below, the binary argument must be provided")
	}
	return res, nil
}

// parse parses, post-processes and verifies the trace. It returns the
// trace version and the list of events.
func parse(r io.Reader, bin string) (int, ParseResult, error) {
	ver, rawEvents, strings, err := readTrace(r)
	if err != nil {
		return 0, ParseResult{}, err
	}
	events, stacks, err := parseEvents(ver, rawEvents, strings)
	if err != nil {
		return 0, ParseResult{}, err
	}
	events = removeFutile(events)
	err = postProcessTrace(ver, events)
	if err != nil {
		return 0, ParseResult{}, err
	}
	// Attach stack traces.
	for _, ev := range events {
		if ev.StkID != 0 {
			ev.Stk = stacks[ev.StkID]
		}
	}
	if ver < 1007 && bin != "" {
		if err := symbolize(events, bin); err != nil {
			return 0, ParseResult{}, err
		}
	}
	return ver, ParseResult{Events: events, Stacks: stacks}, nil
}

// rawEvent is a helper type used during parsing.
type rawEvent struct {
	off   int
	typ   byte
	args  []uint64
	sargs []string
}

// readTrace does wire-format parsing and verification.
// It does not care about specific event types and argument meaning.
func readTrace(r io.Reader) (ver int, events []rawEvent, strings map[uint64]string, err error) {
	// Read and validate trace header.
	var buf [16]byte
	off, err := io.ReadFull(r, buf[:])
	if err != nil {
		err = fmt.Errorf("failed to read header: read %v, err %v", off, err)
		return
	}
	ver, err = parseHeader(buf[:])
	if err != nil {
		return
	}
	switch ver {
	case 1005, 1007, 1008, 1009, 1010, 1011:
		// Note: When adding a new version, add canned traces
		// from the old version to the test suite using mkcanned.bash.
		break
	default:
		err = fmt.Errorf("unsupported trace file version %v.%v (update Go toolchain) %v", ver/1000, ver%1000, ver)
		return
	}

	// Read events.
	strings = make(map[uint64]string)
	for {
		// Read event type and number of arguments (1 byte).
		off0 := off
		var n int
		n, err = r.Read(buf[:1])
		if err == io.EOF {
			err = nil
			break
		}
		if err != nil || n != 1 {
			err = fmt.Errorf("failed to read trace at offset 0x%x: n=%v err=%v", off0, n, err)
			return
		}
		off += n
		typ := buf[0] << 2 >> 2
		narg := buf[0]>>6 + 1
		inlineArgs := byte(4)
		if ver < 1007 {
			narg++
			inlineArgs++
		}
		if typ == EvNone || typ >= EvCount || EventDescriptions[typ].minVersion > ver {
			err = fmt.Errorf("unknown event type %v at offset 0x%x", typ, off0)
			return
		}
		if typ == EvString {
			// String dictionary entry [ID, length, string].
			var id uint64
			id, off, err = readVal(r, off)
			if err != nil {
				return
			}
			if id == 0 {
				err = fmt.Errorf("string at offset %d has invalid id 0", off)
				return
			}
			if strings[id] != "" {
				err = fmt.Errorf("string at offset %d has duplicate id %v", off, id)
				return
			}
			var ln uint64
			ln, off, err = readVal(r, off)
			if err != nil {
				return
			}
			if ln == 0 {
				err = fmt.Errorf("string at offset %d has invalid length 0", off)
				return
			}
			if ln > 1e6 {
				err = fmt.Errorf("string at offset %d has too large length %v", off, ln)
				return
			}
			buf := make([]byte, ln)
			var n int
			n, err = io.ReadFull(r, buf)
			if err != nil {
				err = fmt.Errorf("failed to read trace at offset %d: read %v, want %v, error %v", off, n, ln, err)
				return
			}
			off += n
			strings[id] = string(buf)
			continue
		}
		ev := rawEvent{typ: typ, off: off0}
		if narg < inlineArgs {
			for i := 0; i < int(narg); i++ {
				var v uint64
				v, off, err = readVal(r, off)
				if err != nil {
					err = fmt.Errorf("failed to read event %v argument at offset %v (%v)", typ, off, err)
					return
				}
				ev.args = append(ev.args, v)
			}
		} else {
			// More than inlineArgs args, the first value is length of the event in bytes.
			var v uint64
			v, off, err = readVal(r, off)
			if err != nil {
				err = fmt.Errorf("failed to read event %v argument at offset %v (%v)", typ, off, err)
				return
			}
			evLen := v
			off1 := off
			for evLen > uint64(off-off1) {
				v, off, err = readVal(r, off)
				if err != nil {
					err = fmt.Errorf("failed to read event %v argument at offset %v (%v)", typ, off, err)
					return
				}
				ev.args = append(ev.args, v)
			}
			if evLen != uint64(off-off1) {
				err = fmt.Errorf("event has wrong length at offset 0x%x: want %v, got %v", off0, evLen, off-off1)
				return
			}
		}
		switch ev.typ {
		case EvUserLog: // EvUserLog records are followed by a value string of length ev.args[len(ev.args)-1]
			var s string
			s, off, err = readStr(r, off)
			ev.sargs = append(ev.sargs, s)
		}
		events = append(events, ev)
	}
	return
}

func readStr(r io.Reader, off0 int) (s string, off int, err error) {
	var sz uint64
	sz, off, err = readVal(r, off0)
	if err != nil || sz == 0 {
		return "", off, err
	}
	if sz > 1e6 {
		return "", off, fmt.Errorf("string at offset %d is too large (len=%d)", off, sz)
	}
	buf := make([]byte, sz)
	n, err := io.ReadFull(r, buf)
	if err != nil || sz != uint64(n) {
		return "", off + n, fmt.Errorf("failed to read trace at offset %d: read %v, want %v, error %v", off, n, sz, err)
	}
	return string(buf), off + n, nil
}

// parseHeader parses trace header of the form "go 1.7 trace\x00\x00\x00\x00"
// and returns parsed version as 1007.
func parseHeader(buf []byte) (int, error) {
	if len(buf) != 16 {
		return 0, fmt.Errorf("bad header length")
	}
	if buf[0] != 'g' || buf[1] != 'o' || buf[2] != ' ' ||
		buf[3] < '1' || buf[3] > '9' ||
		buf[4] != '.' ||
		buf[5] < '1' || buf[5] > '9' {
		return 0, fmt.Errorf("not a trace file")
	}
	ver := int(buf[5] - '0')
	i := 0
	for ; buf[6+i] >= '0' && buf[6+i] <= '9' && i < 2; i++ {
		ver = ver*10 + int(buf[6+i]-'0')
	}
	ver += int(buf[3]-'0') * 1000
	if !bytes.Equal(buf[6+i:], []byte(" trace\x00\x00\x00\x00")[:10-i]) {
		return 0, fmt.Errorf("not a trace file")
	}
	return ver, nil
}

// Parse events transforms raw events into events.
// It does analyze and verify per-event-type arguments.
func parseEvents(ver int, rawEvents []rawEvent, strings map[uint64]string) (events []*Event, stacks map[uint64][]*Frame, err error) {
	var ticksPerSec, lastSeq, lastTs int64
	var lastG uint64
	var lastP int
	timerGoids := make(map[uint64]bool)
	lastGs := make(map[int]uint64) // last goroutine running on P
	stacks = make(map[uint64][]*Frame)
	batches := make(map[int][]*Event) // events by P
	for _, raw := range rawEvents {
		desc := EventDescriptions[raw.typ]
		if desc.Name == "" {
			err = fmt.Errorf("missing description for event type %v", raw.typ)
			return
		}
		narg := argNum(raw, ver)
		if len(raw.args) != narg {
			err = fmt.Errorf("%v has wrong number of arguments at offset 0x%x: want %v, got %v",
				desc.Name, raw.off, narg, len(raw.args))
			return
		}
		switch raw.typ {
		case EvBatch:
			lastGs[lastP] = lastG
			lastP = int(raw.args[0])
			lastG = lastGs[lastP]
			if ver < 1007 {
				lastSeq = int64(raw.args[1])
				lastTs = int64(raw.args[2])
			} else {
				lastTs = int64(raw.args[1])
			}
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
			timerGoids[raw.args[0]] = true
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
			want := 2 + 4*size
			if ver < 1007 {
				want = 2 + size
			}
			if uint64(len(raw.args)) != want {
				err = fmt.Errorf("EvStack has wrong number of arguments at offset 0x%x: want %v, got %v",
					raw.off, want, len(raw.args))
				return
			}
			id := raw.args[0]
			if id != 0 && size > 0 {
				stk := make([]*Frame, size)
				for i := 0; i < int(size); i++ {
					if ver < 1007 {
						stk[i] = &Frame{PC: raw.args[2+i]}
					} else {
						pc := raw.args[2+i*4+0]
						fn := raw.args[2+i*4+1]
						file := raw.args[2+i*4+2]
						line := raw.args[2+i*4+3]
						stk[i] = &Frame{PC: pc, Fn: strings[fn], File: strings[file], Line: int(line)}
					}
				}
				stacks[id] = stk
			}
		default:
			e := &Event{Off: raw.off, Type: raw.typ, P: lastP, G: lastG}
			var argOffset int
			if ver < 1007 {
				e.seq = lastSeq + int64(raw.args[0])
				e.Ts = lastTs + int64(raw.args[1])
				lastSeq = e.seq
				argOffset = 2
			} else {
				e.Ts = lastTs + int64(raw.args[0])
				argOffset = 1
			}
			lastTs = e.Ts
			for i := argOffset; i < narg; i++ {
				if i == narg-1 && desc.Stack {
					e.StkID = raw.args[i]
				} else {
					e.Args[i-argOffset] = raw.args[i]
				}
			}
			switch raw.typ {
			case EvGoStart, EvGoStartLocal, EvGoStartLabel:
				lastG = e.Args[0]
				e.G = lastG
				if raw.typ == EvGoStartLabel {
					e.SArgs = []string{strings[e.Args[2]]}
				}
			case EvGCSTWStart:
				e.G = 0
				switch e.Args[0] {
				case 0:
					e.SArgs = []string{"mark termination"}
				case 1:
					e.SArgs = []string{"sweep termination"}
				default:
					err = fmt.Errorf("unknown STW kind %d", e.Args[0])
					return
				}
			case EvGCStart, EvGCDone, EvGCSTWDone:
				e.G = 0
			case EvGoEnd, EvGoStop, EvGoSched, EvGoPreempt,
				EvGoSleep, EvGoBlock, EvGoBlockSend, EvGoBlockRecv,
				EvGoBlockSelect, EvGoBlockSync, EvGoBlockCond, EvGoBlockNet,
				EvGoSysBlock, EvGoBlockGC:
				lastG = 0
			case EvGoSysExit, EvGoWaiting, EvGoInSyscall:
				e.G = e.Args[0]
			case EvUserTaskCreate:
				// e.Args 0: taskID, 1:parentID, 2:nameID
				e.SArgs = []string{strings[e.Args[2]]}
			case EvUserRegion:
				// e.Args 0: taskID, 1: mode, 2:nameID
				e.SArgs = []string{strings[e.Args[2]]}
			case EvUserLog:
				// e.Args 0: taskID, 1:keyID, 2: stackID
				e.SArgs = []string{strings[e.Args[1]], raw.sargs[0]}
			}
			batches[lastP] = append(batches[lastP], e)
		}
	}
	if len(batches) == 0 {
		err = fmt.Errorf("trace is empty")
		return
	}
	if ticksPerSec == 0 {
		err = fmt.Errorf("no EvFrequency event")
		return
	}
	if BreakTimestampsForTesting {
		var batchArr [][]*Event
		for _, batch := range batches {
			batchArr = append(batchArr, batch)
		}
		for i := 0; i < 5; i++ {
			batch := batchArr[rand.Intn(len(batchArr))]
			batch[rand.Intn(len(batch))].Ts += int64(rand.Intn(2000) - 1000)
		}
	}
	if ver < 1007 {
		events, err = order1005(batches)
	} else {
		events, err = order1007(batches)
	}
	if err != nil {
		return
	}

	// Translate cpu ticks to real time.
	minTs := events[0].Ts
	// Use floating point to avoid integer overflows.
	freq := 1e9 / float64(ticksPerSec)
	for _, ev := range events {
		ev.Ts = int64(float64(ev.Ts-minTs) * freq)
		// Move timers and syscalls to separate fake Ps.
		if timerGoids[ev.G] && ev.Type == EvGoUnblock {
			ev.P = TimerP
		}
		if ev.Type == EvGoSysExit {
			ev.P = SyscallP
		}
	}

	return
}

// removeFutile removes all constituents of futile wakeups (block, unblock, start).
// For example, a goroutine was unblocked on a mutex, but another goroutine got
// ahead and acquired the mutex before the first goroutine is scheduled,
// so the first goroutine has to block again. Such wakeups happen on buffered
// channels and sync.Mutex, but are generally not interesting for end user.
func removeFutile(events []*Event) []*Event {
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
	return newEvents
}

// ErrTimeOrder is returned by Parse when the trace contains
// time stamps that do not respect actual event ordering.
var ErrTimeOrder = fmt.Errorf("time stamps out of order")

// postProcessTrace does inter-event verification and information restoration.
// The resulting trace is guaranteed to be consistent
// (for example, a P does not run two Gs at the same time, or a G is indeed
// blocked before an unblock event).
func postProcessTrace(ver int, events []*Event) error {
	const (
		gDead = iota
		gRunnable
		gRunning
		gWaiting
	)
	type gdesc struct {
		state        int
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
	activeRegions := make(map[uint64][]*Event) // goroutine id to stack of regions
	gs[0] = gdesc{state: gRunning}
	var evGC, evSTW *Event

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
			// Attribute this to the global GC state.
			ev.P = GCP
		case EvGCDone:
			if evGC == nil {
				return fmt.Errorf("bogus GC end (offset %v, time %v)", ev.Off, ev.Ts)
			}
			evGC.Link = ev
			evGC = nil
		case EvGCSTWStart:
			evp := &evSTW
			if ver < 1010 {
				// Before 1.10, EvGCSTWStart was per-P.
				evp = &p.evSTW
			}
			if *evp != nil {
				return fmt.Errorf("previous STW is not ended before a new one (offset %v, time %v)", ev.Off, ev.Ts)
			}
			*evp = ev
		case EvGCSTWDone:
			evp := &evSTW
			if ver < 1010 {
				// Before 1.10, EvGCSTWDone was per-P.
				evp = &p.evSTW
			}
			if *evp == nil {
				return fmt.Errorf("bogus STW end (offset %v, time %v)", ev.Off, ev.Ts)
			}
			(*evp).Link = ev
			*evp = nil
		case EvGCSweepStart:
			if p.evSweep != nil {
				return fmt.Errorf("previous sweeping is not ended before a new one (offset %v, time %v)", ev.Off, ev.Ts)
			}
			p.evSweep = ev
		case EvGCMarkAssistStart:
			if g.evMarkAssist != nil {
				return fmt.Errorf("previous mark assist is not ended before a new one (offset %v, time %v)", ev.Off, ev.Ts)
			}
			g.evMarkAssist = ev
		case EvGCMarkAssistDone:
			// Unlike most events, mark assists can be in progress when a
			// goroutine starts tracing, so we can't report an error here.
			if g.evMarkAssist != nil {
				g.evMarkAssist.Link = ev
				g.evMarkAssist = nil
			}
		case EvGCSweepDone:
			if p.evSweep == nil {
				return fmt.Errorf("bogus sweeping end (offset %v, time %v)", ev.Off, ev.Ts)
			}
			p.evSweep.Link = ev
			p.evSweep = nil
		case EvGoWaiting:
			if g.state != gRunnable {
				return fmt.Errorf("g %v is not runnable before EvGoWaiting (offset %v, time %v)", ev.G, ev.Off, ev.Ts)
			}
			g.state = gWaiting
			g.ev = ev
		case EvGoInSyscall:
			if g.state != gRunnable {
				return fmt.Errorf("g %v is not runnable before EvGoInSyscall (offset %v, time %v)", ev.G, ev.Off, ev.Ts)
			}
			g.state = gWaiting
			g.ev = ev
		case EvGoCreate:
			if err := checkRunning(p, g, ev, true); err != nil {
				return err
			}
			if _, ok := gs[ev.Args[0]]; ok {
				return fmt.Errorf("g %v already exists (offset %v, time %v)", ev.Args[0], ev.Off, ev.Ts)
			}
			gs[ev.Args[0]] = gdesc{state: gRunnable, ev: ev, evCreate: ev}
		case EvGoStart, EvGoStartLabel:
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
				if ver < 1007 {
					// +1 because symbolizer expects return pc.
					ev.Stk = []*Frame{{PC: g.evCreate.Args[1] + 1}}
				} else {
					ev.StkID = g.evCreate.Args[1]
				}
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

			if ev.Type == EvGoEnd { // flush all active regions
				regions := activeRegions[ev.G]
				for _, s := range regions {
					s.Link = ev
				}
				delete(activeRegions, ev.G)
			}

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
			EvGoBlockSelect, EvGoBlockSync, EvGoBlockCond, EvGoBlockNet, EvGoBlockGC:
			if err := checkRunning(p, g, ev, false); err != nil {
				return err
			}
			g.state = gWaiting
			g.ev = ev
			g.evStart.Link = ev
			g.evStart = nil
			p.g = 0
		case EvUserTaskCreate:
			taskid := ev.Args[0]
			if prevEv, ok := tasks[taskid]; ok {
				return fmt.Errorf("task id conflicts (id:%d), %q vs %q", taskid, ev, prevEv)
			}
			tasks[ev.Args[0]] = ev
		case EvUserTaskEnd:
			taskid := ev.Args[0]
			if taskCreateEv, ok := tasks[taskid]; ok {
				taskCreateEv.Link = ev
				delete(tasks, taskid)
			}
		case EvUserRegion:
			mode := ev.Args[1]
			regions := activeRegions[ev.G]
			if mode == 0 { // region start
				activeRegions[ev.G] = append(regions, ev) // push
			} else if mode == 1 { // region end
				n := len(regions)
				if n > 0 { // matching region start event is in the trace.
					s := regions[n-1]
					if s.Args[0] != ev.Args[0] || s.SArgs[0] != ev.SArgs[0] { // task id, region name mismatch
						return fmt.Errorf("misuse of region in goroutine %d: span end %q when the inner-most active span start event is %q", ev.G, ev, s)
					}
					// Link region start event with span end event
					s.Link = ev

					if n > 1 {
						activeRegions[ev.G] = regions[:n-1]
					} else {
						delete(activeRegions, ev.G)
					}
				}
			} else {
				return fmt.Errorf("invalid user region mode: %q", ev)
			}
		}

		gs[ev.G] = g
		ps[ev.P] = p
	}

	// TODO(dvyukov): restore stacks for EvGoStart events.
	// TODO(dvyukov): test that all EvGoStart events has non-nil Link.

	return nil
}

// symbolize attaches func/file/line info to stack traces.
func symbolize(events []*Event, bin string) error {
	// First, collect and dedup all pcs.
	pcs := make(map[uint64]*Frame)
	for _, ev := range events {
		for _, f := range ev.Stk {
			pcs[f.PC] = nil
		}
	}

	// Start addr2line.
	cmd := exec.Command(goCmd(), "tool", "addr2line", bin)
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

// Print dumps events to stdout. For debugging.
func Print(events []*Event) {
	for _, ev := range events {
		PrintEvent(ev)
	}
}

// PrintEvent dumps the event to stdout. For debugging.
func PrintEvent(ev *Event) {
	fmt.Printf("%s\n", ev)
}

func (ev *Event) String() string {
	desc := EventDescriptions[ev.Type]
	w := new(bytes.Buffer)
	fmt.Fprintf(w, "%v %v p=%v g=%v off=%v", ev.Ts, desc.Name, ev.P, ev.G, ev.Off)
	for i, a := range desc.Args {
		fmt.Fprintf(w, " %v=%v", a, ev.Args[i])
	}
	for i, a := range desc.SArgs {
		fmt.Fprintf(w, " %v=%v", a, ev.SArgs[i])
	}
	return w.String()
}

// argNum returns total number of args for the event accounting for timestamps,
// sequence numbers and differences between trace format versions.
func argNum(raw rawEvent, ver int) int {
	desc := EventDescriptions[raw.typ]
	if raw.typ == EvStack {
		return len(raw.args)
	}
	narg := len(desc.Args)
	if desc.Stack {
		narg++
	}
	switch raw.typ {
	case EvBatch, EvFrequency, EvTimerGoroutine:
		if ver < 1007 {
			narg++ // there was an unused arg before 1.7
		}
		return narg
	}
	narg++ // timestamp
	if ver < 1007 {
		narg++ // sequence
	}
	switch raw.typ {
	case EvGCSweepDone:
		if ver < 1009 {
			narg -= 2 // 1.9 added two arguments
		}
	case EvGCStart, EvGoStart, EvGoUnblock:
		if ver < 1007 {
			narg-- // 1.7 added an additional seq arg
		}
	case EvGCSTWStart:
		if ver < 1010 {
			narg-- // 1.10 added an argument
		}
	}
	return narg
}

// BreakTimestampsForTesting causes the parser to randomly alter timestamps (for testing of broken cputicks).
var BreakTimestampsForTesting bool

// Event types in the trace.
// Verbatim copy from src/runtime/trace.go with the "trace" prefix removed.
const (
	EvNone              = 0  // unused
	EvBatch             = 1  // start of per-P batch of events [pid, timestamp]
	EvFrequency         = 2  // contains tracer timer frequency [frequency (ticks per second)]
	EvStack             = 3  // stack [stack id, number of PCs, array of {PC, func string ID, file string ID, line}]
	EvGomaxprocs        = 4  // current value of GOMAXPROCS [timestamp, GOMAXPROCS, stack id]
	EvProcStart         = 5  // start of P [timestamp, thread id]
	EvProcStop          = 6  // stop of P [timestamp]
	EvGCStart           = 7  // GC start [timestamp, seq, stack id]
	EvGCDone            = 8  // GC done [timestamp]
	EvGCSTWStart        = 9  // GC mark termination start [timestamp, kind]
	EvGCSTWDone         = 10 // GC mark termination done [timestamp]
	EvGCSweepStart      = 11 // GC sweep start [timestamp, stack id]
	EvGCSweepDone       = 12 // GC sweep done [timestamp, swept, reclaimed]
	EvGoCreate          = 13 // goroutine creation [timestamp, new goroutine id, new stack id, stack id]
	EvGoStart           = 14 // goroutine starts running [timestamp, goroutine id, seq]
	EvGoEnd             = 15 // goroutine ends [timestamp]
	EvGoStop            = 16 // goroutine stops (like in select{}) [timestamp, stack]
	EvGoSched           = 17 // goroutine calls Gosched [timestamp, stack]
	EvGoPreempt         = 18 // goroutine is preempted [timestamp, stack]
	EvGoSleep           = 19 // goroutine calls Sleep [timestamp, stack]
	EvGoBlock           = 20 // goroutine blocks [timestamp, stack]
	EvGoUnblock         = 21 // goroutine is unblocked [timestamp, goroutine id, seq, stack]
	EvGoBlockSend       = 22 // goroutine blocks on chan send [timestamp, stack]
	EvGoBlockRecv       = 23 // goroutine blocks on chan recv [timestamp, stack]
	EvGoBlockSelect     = 24 // goroutine blocks on select [timestamp, stack]
	EvGoBlockSync       = 25 // goroutine blocks on Mutex/RWMutex [timestamp, stack]
	EvGoBlockCond       = 26 // goroutine blocks on Cond [timestamp, stack]
	EvGoBlockNet        = 27 // goroutine blocks on network [timestamp, stack]
	EvGoSysCall         = 28 // syscall enter [timestamp, stack]
	EvGoSysExit         = 29 // syscall exit [timestamp, goroutine id, seq, real timestamp]
	EvGoSysBlock        = 30 // syscall blocks [timestamp]
	EvGoWaiting         = 31 // denotes that goroutine is blocked when tracing starts [timestamp, goroutine id]
	EvGoInSyscall       = 32 // denotes that goroutine is in syscall when tracing starts [timestamp, goroutine id]
	EvHeapAlloc         = 33 // gcController.heapLive change [timestamp, heap live bytes]
	EvHeapGoal          = 34 // gcController.heapGoal change [timestamp, heap goal bytes]
	EvTimerGoroutine    = 35 // denotes timer goroutine [timer goroutine id]
	EvFutileWakeup      = 36 // denotes that the previous wakeup of this goroutine was futile [timestamp]
	EvString            = 37 // string dictionary entry [ID, length, string]
	EvGoStartLocal      = 38 // goroutine starts running on the same P as the last event [timestamp, goroutine id]
	EvGoUnblockLocal    = 39 // goroutine is unblocked on the same P as the last event [timestamp, goroutine id, stack]
	EvGoSysExitLocal    = 40 // syscall exit on the same P as the last event [timestamp, goroutine id, real timestamp]
	EvGoStartLabel      = 41 // goroutine starts running with label [timestamp, goroutine id, seq, label string id]
	EvGoBlockGC         = 42 // goroutine blocks on GC assist [timestamp, stack]
	EvGCMarkAssistStart = 43 // GC mark assist start [timestamp, stack]
	EvGCMarkAssistDone  = 44 // GC mark assist done [timestamp]
	EvUserTaskCreate    = 45 // trace.NewContext [timestamp, internal task id, internal parent id, stack, name string]
	EvUserTaskEnd       = 46 // end of task [timestamp, internal task id, stack]
	EvUserRegion        = 47 // trace.WithRegion [timestamp, internal task id, mode(0:start, 1:end), stack, name string]
	EvUserLog           = 48 // trace.Log [timestamp, internal id, key string id, stack, value string]
	EvCount             = 49
)

var EventDescriptions = [EvCount]struct {
	Name       string
	minVersion int
	Stack      bool
	Args       []string
	SArgs      []string // string arguments
}{
	EvNone:              {"None", 1005, false, []string{}, nil},
	EvBatch:             {"Batch", 1005, false, []string{"p", "ticks"}, nil}, // in 1.5 format it was {"p", "seq", "ticks"}
	EvFrequency:         {"Frequency", 1005, false, []string{"freq"}, nil},   // in 1.5 format it was {"freq", "unused"}
	EvStack:             {"Stack", 1005, false, []string{"id", "siz"}, nil},
	EvGomaxprocs:        {"Gomaxprocs", 1005, true, []string{"procs"}, nil},
	EvProcStart:         {"ProcStart", 1005, false, []string{"thread"}, nil},
	EvProcStop:          {"ProcStop", 1005, false, []string{}, nil},
	EvGCStart:           {"GCStart", 1005, true, []string{"seq"}, nil}, // in 1.5 format it was {}
	EvGCDone:            {"GCDone", 1005, false, []string{}, nil},
	EvGCSTWStart:        {"GCSTWStart", 1005, false, []string{"kindid"}, []string{"kind"}}, // <= 1.9, args was {} (implicitly {0})
	EvGCSTWDone:         {"GCSTWDone", 1005, false, []string{}, nil},
	EvGCSweepStart:      {"GCSweepStart", 1005, true, []string{}, nil},
	EvGCSweepDone:       {"GCSweepDone", 1005, false, []string{"swept", "reclaimed"}, nil}, // before 1.9, format was {}
	EvGoCreate:          {"GoCreate", 1005, true, []string{"g", "stack"}, nil},
	EvGoStart:           {"GoStart", 1005, false, []string{"g", "seq"}, nil}, // in 1.5 format it was {"g"}
	EvGoEnd:             {"GoEnd", 1005, false, []string{}, nil},
	EvGoStop:            {"GoStop", 1005, true, []string{}, nil},
	EvGoSched:           {"GoSched", 1005, true, []string{}, nil},
	EvGoPreempt:         {"GoPreempt", 1005, true, []string{}, nil},
	EvGoSleep:           {"GoSleep", 1005, true, []string{}, nil},
	EvGoBlock:           {"GoBlock", 1005, true, []string{}, nil},
	EvGoUnblock:         {"GoUnblock", 1005, true, []string{"g", "seq"}, nil}, // in 1.5 format it was {"g"}
	EvGoBlockSend:       {"GoBlockSend", 1005, true, []string{}, nil},
	EvGoBlockRecv:       {"GoBlockRecv", 1005, true, []string{}, nil},
	EvGoBlockSelect:     {"GoBlockSelect", 1005, true, []string{}, nil},
	EvGoBlockSync:       {"GoBlockSync", 1005, true, []string{}, nil},
	EvGoBlockCond:       {"GoBlockCond", 1005, true, []string{}, nil},
	EvGoBlockNet:        {"GoBlockNet", 1005, true, []string{}, nil},
	EvGoSysCall:         {"GoSysCall", 1005, true, []string{}, nil},
	EvGoSysExit:         {"GoSysExit", 1005, false, []string{"g", "seq", "ts"}, nil},
	EvGoSysBlock:        {"GoSysBlock", 1005, false, []string{}, nil},
	EvGoWaiting:         {"GoWaiting", 1005, false, []string{"g"}, nil},
	EvGoInSyscall:       {"GoInSyscall", 1005, false, []string{"g"}, nil},
	EvHeapAlloc:         {"HeapAlloc", 1005, false, []string{"mem"}, nil},
	EvHeapGoal:          {"HeapGoal", 1005, false, []string{"mem"}, nil},
	EvTimerGoroutine:    {"TimerGoroutine", 1005, false, []string{"g"}, nil}, // in 1.5 format it was {"g", "unused"}
	EvFutileWakeup:      {"FutileWakeup", 1005, false, []string{}, nil},
	EvString:            {"String", 1007, false, []string{}, nil},
	EvGoStartLocal:      {"GoStartLocal", 1007, false, []string{"g"}, nil},
	EvGoUnblockLocal:    {"GoUnblockLocal", 1007, true, []string{"g"}, nil},
	EvGoSysExitLocal:    {"GoSysExitLocal", 1007, false, []string{"g", "ts"}, nil},
	EvGoStartLabel:      {"GoStartLabel", 1008, false, []string{"g", "seq", "labelid"}, []string{"label"}},
	EvGoBlockGC:         {"GoBlockGC", 1008, true, []string{}, nil},
	EvGCMarkAssistStart: {"GCMarkAssistStart", 1009, true, []string{}, nil},
	EvGCMarkAssistDone:  {"GCMarkAssistDone", 1009, false, []string{}, nil},
	EvUserTaskCreate:    {"UserTaskCreate", 1011, true, []string{"taskid", "pid", "typeid"}, []string{"name"}},
	EvUserTaskEnd:       {"UserTaskEnd", 1011, true, []string{"taskid"}, nil},
	EvUserRegion:        {"UserRegion", 1011, true, []string{"taskid", "mode", "typeid"}, []string{"name"}},
	EvUserLog:           {"UserLog", 1011, true, []string{"id", "keyid"}, []string{"category", "message"}},
}

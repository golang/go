// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package traceparser parses the trace files produced by runtime.StartTrace
package traceparser

import (
	"fmt"
	"internal/traceparser/filebuf"
	"io"
	"strings"
)

// Parsed is the result of parsing a trace file
type Parsed struct {
	// Set by New
	Name               string // File's name
	Size               int64  // File's size
	Count              int64  // approximate number of all events
	MaxTs              int64  // range of all events, in nanoseconds
	Strings            map[uint64]string
	Stacks             map[uint32][]*Frame
	Version            int         // version of the trace file from header
	TicksPerSec        int64       // from EvFrequency in trailer
	minticks, maxticks int64       // from Init
	r                  filebuf.Buf // implements io.Seek and io.Read
	batches            []batch     // location of each Batch and time of start
	timerGoids         map[uint64]bool
	// the following are per Parse
	MinWant, MaxWant int64    // range wanted, from the arguments to Parse()
	Err              error    // set by internal functions to stop further processing
	Events           []*Event // after Parse, the events from MinWant to MaxWant
	Preflen          int      // how long a prefix we added
	Ignored          int      // how many events we elided
	Added            int      // how many events we added, not including the prefix
	// internal processing variables
	seenArgs map[uint64]*[]uint64
	byproc   map[int][]*Event
	lastGs   map[int]uint64
	lastG    uint64
	lastP    int
	lastTs   int64
}

func (p *Parsed) String() string {
	ans := []string{}
	ans = append(ans, fmt.Sprintf("%s Sz:%d Count:%d MaxTs:%d #strs:%d #stks:%d",
		p.Name, p.Size, p.Count, p.MaxTs, len(p.Strings), len(p.Stacks)))
	ans = append(ans, fmt.Sprintf("%d clock:%d ticks:(%d,%d) #b:%d",
		p.Version, p.TicksPerSec, p.minticks, p.maxticks, len(p.batches)))
	return strings.Join(ans, "\n\t")
}

// clean up after previous call to Parse
func (p *Parsed) clean() {
	// some of these are redundant
	p.Err = nil
	p.Events = nil
	p.Preflen = 0
	p.Ignored = 0
	p.Added = 0
	p.seenArgs = nil // redundant, but safe
	p.byproc = nil
	p.lastGs = nil
	p.lastG = 0
	p.lastTs = 0
}

// Frame is a frame in a stack traces
type Frame struct {
	PC   uint64
	Fn   string
	File string
	Line int
}

// An Event is the parsed form of a single trace event
type Event struct {
	Type  byte
	P     int32
	Ts    int64
	G     uint64
	StkID uint32 // key to Parsed.Stacks
	Args  [3]uint64
	SArgs []string // EvUserLog has 2. Others are 1 or none
	Link  *Event
}

// Batch remembers the EvBatch events. PJW: keep an index of User events?
type batch struct {
	Off    int
	P      int64
	Cycles int64      // as read from EvBatch
	Nano   int64      // start time of batch, set in commonInit()
	raws   []rawEvent // filled in during Parse() for those batches that overlap the desired interval
}

// rawEvent is a raw event parsed from batches that overlap the time interval
type rawEvent struct { // about 75 bytes
	// the choice of what to share (args) and what to make unique per rawEvent
	// (arg0, sarg) was done by measuring the space impact of various choices.
	off  uint32 // offset in batch (at batch.Off + off in file)
	typ  byte
	arg0 uint64
	args *[]uint64 // remainder of the args (frequently nil), shared
	sarg string
}

func (r rawEvent) String() string {
	if r.args != nil && len(*r.args) > 0 {
		return fmt.Sprintf("[%s %d %v %s]", evname(r.typ), r.arg0, *r.args, r.sarg)
	}
	return fmt.Sprintf("[%s, %d, [], %s]", evname(r.typ), r.arg0, r.sarg)
}

// New scans the trace file, finding the number of events, the earliest and latest
// timestamps, and the stacks and strings referenced in the file.
func New(fname string) (*Parsed, error) {
	fd, err := filebuf.New(fname)
	if err != nil {
		return nil, err
	}
	return commonInit(fd, fname)
}

// ParseError may be returned by New() or ParseBuffer() to make available
// some information in the case that the raw trace file seems to contain
// negative time stamps. (In P, Name, Size, count, Strings, Stacks, Versions are valid,
// and MaxTs or TicksPerSec is negative.)
type ParseError struct {
	P   *Parsed
	Err error
}

func (pe ParseError) Error() string {
	return pe.Err.Error()
}

func commonInit(fd filebuf.Buf, fname string) (*Parsed, error) {
	ans := &Parsed{Name: fname, minticks: 1 << 62} // minticks can only decrease
	var err error
	defer func() {
		if err != nil {
			fd.Close() // try to clean up after error
		}
	}()
	ans.Size = fd.Size()
	ans.r = fd
	// parseRaw here for header, trailer: clock, stacks, strings,
	if err = ans.parseHeader(); err != nil {
		return nil, err
	}
	if err = ans.scanFile(); err != nil {
		return nil, err
	}
	// done with seenArgs
	ans.seenArgs = nil
	// convert the clicks in batches to nanoseconds
	ans.toNanoseconds()
	if ans.MaxTs <= 0 || ans.TicksPerSec <= 0 {
		err := ParseError{
			P: ans,
			Err: fmt.Errorf("corrupt trace file: negative time: (max TS=%d, ticks per sec=%d",
				ans.MaxTs, ans.TicksPerSec),
		}
		return nil, err
	}
	return ans, nil
}

// Parse parses the events in the interval: start <= ts <= start+length.
// f, if not nil, will be called at various stages of the parse, each identified by the string
// argument. It could report on elapsed time, or memory usage, or whatever the user wants.
// The number of times it is called and the contents of the string argument are both
// changeable details of the implementation. Parse is not safe for concurrent use.
func (p *Parsed) Parse(start, length int64, f func(string)) error {
	p.clean()
	if f == nil {
		f = func(string) {} // avoid any further testing for nil
	}

	p.MinWant = start
	p.MaxWant = start + length
	// arrange the slice of batches by P
	byp := map[int64][]*batch{}
	// PJW: keep track of the order the Ps occur and use that for batchify
	for i, b := range p.batches {
		byp[b.P] = append(byp[b.P], &p.batches[i])
		p.batches[i].raws = nil // reset from last call to Parse
	}
	// batchify the ones that overlap the time range
	for _, v := range byp {
		for i := 0; i < len(v); i++ {
			b := v[i]
			var bnext *batch
			if i < len(v)-1 {
				bnext = v[i+1]
			}
			if b.Nano >= p.MaxWant {
				// starts too late
				continue
			} else if b.Nano <= p.MinWant && (bnext != nil && bnext.Nano <= p.MinWant) {
				// entirely too early
				continue
			}
			err := p.batchify(b)
			if err != nil {
				return err
			}
		}
	}
	f("batchify done")
	return p.createEvents(f)
}

// ParseBuffer treats its argument as a trace file, and returns the
// result of parsing it
func ParseBuffer(rd io.Reader) (*Parsed, error) {
	pr, err := filebuf.FromReader(rd)
	if err != nil {
		return nil, err
	}
	p, err := commonInit(pr, "<buf>")
	if err != nil {
		return nil, err
	}
	// need the version and the initial scan
	err = p.Parse(0, 1<<62, nil)
	if err != nil {
		return nil, err
	}
	return p, nil
}

// called from commonInit to compute the nanosecond when batches start
func (p *Parsed) toNanoseconds() {
	minCycles := p.minticks
	freq := 1e9 / float64(p.TicksPerSec)
	// Batches, and more to come.  Don't call this twice!
	for i, ex := range p.batches {
		p.batches[i].Nano = int64(float64(ex.Cycles-minCycles) * freq)
	}
	p.MaxTs = int64(float64(p.maxticks-minCycles) * freq)
}

// argsAt returns the args of an event in the file and the offset for the next event.
//
// For EvString it returns off, nil, nil, and
// for EvUserLog it ignores the string argument, which must be read by the
// caller.
func (p *Parsed) argsAt(off int, check byte) (int, []uint64, error) {
	off0 := off
	r := p.r
	loc, err := r.Seek(int64(off), 0)
	if err != nil {
		panic(err)
	}
	var buf [1]byte
	n, err := r.Read(buf[:])
	if err != nil || n != 1 {
		return 0, nil, fmt.Errorf("read failed at 0x%x, %d %v, loc=%d",
			off, n, err, loc)
	}
	off += n
	typ := buf[0] << 2 >> 2
	narg := buf[0]>>6 + 1
	inlineArgs := byte(4)

	if typ == EvNone || typ >= EvCount ||
		EventDescriptions[typ].MinVersion > p.Version {
		return 0, nil, fmt.Errorf("unk type %v at offset 0x%x", typ, off0)
	}
	if typ == EvString { // skip, wihtout error checking
		_, off, err = readVal(r, off)
		var ln uint64
		ln, off, err = readVal(r, off)
		off += int(ln)
		return off, nil, nil
	}
	args := []uint64{}
	if narg < inlineArgs {
		for i := 0; i < int(narg); i++ {
			var v uint64
			v, off, err = readVal(r, off)
			if err != nil {
				err = fmt.Errorf("failed to read event %v argument at offset %v (%v)", typ, off, err)
				return 0, nil, err
			}
			args = append(args, v)
		}
	} else {
		// More than inlineArgs args, the first value is length of the event in bytes.
		var v uint64
		v, off, err = readVal(r, off)
		if err != nil {
			err = fmt.Errorf("failed to read event %v argument at offset %v (%v)", typ, off, err)
			return 0, nil, err
		}
		evLen := v
		off1 := off
		for evLen > uint64(off-off1) {
			v, off, err = readVal(r, off)
			if err != nil {
				err = fmt.Errorf("failed to read event %v argument at offset %v (%v)", typ, off, err)
				return 0, nil, err
			}
			args = append(args, v)
		}
		if evLen != uint64(off-off1) {
			err = fmt.Errorf("event has wrong length at offset 0x%x: want %v, got %v", off0, evLen, off-off1)
			return 0, nil, err
		}
	}
	// This routine does not read the string argument. Callers must tread EvUserLog specially.
	return off, args, nil
}

// read a string from r
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

// OSStats reports on the underlying i/o. If p was created by New,
// the fields report filesystem activity. If p was created by ParseBuffer,
// only Size is set.
func (p *Parsed) OSStats() filebuf.Stat {
	return p.r.Stats()
}

func (ev *Event) String() string {
	var tslink int64
	if ev.Link != nil {
		tslink = ev.Link.Ts
	}
	return fmt.Sprintf("[g:%d p:%d %s/%d %v %v %x ->%x]",
		ev.G, ev.P, evname(ev.Type), ev.Type,
		ev.Args, ev.SArgs, ev.Ts, tslink)

}

func evname(t byte) string {
	if t >= EvCount || t < 0 {
		return fmt.Sprintf("typ%d?", t)
	}
	return EventDescriptions[t].Name
}

// Close the underlying file.
func (p *Parsed) Close() error {
	return p.r.Close()
}

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
	EvHeapAlloc         = 33 // memstats.heap_live change [timestamp, heap_alloc]
	EvNextGC            = 34 // memstats.next_gc change [timestamp, next_gc]
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
	EvUserRegion        = 47 // trace.WithSpan [timestamp, internal task id, mode(0:start, 1:end), stack, name string]
	EvUserLog           = 48 // trace.Log [timestamp, internal id, key string id, stack, value string]
	EvCount             = 49
)

// EventDescriptions describe the Events
var EventDescriptions = [EvCount]struct {
	Name       string
	MinVersion int
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
	EvNextGC:            {"NextGC", 1005, false, []string{"mem"}, nil},
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

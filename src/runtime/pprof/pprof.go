// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package pprof writes runtime profiling data in the format expected
// by the pprof visualization tool.
//
// Profiling a Go program
//
// The first step to profiling a Go program is to enable profiling.
// Support for profiling benchmarks built with the standard testing
// package is built into go test. For example, the following command
// runs benchmarks in the current directory and writes the CPU and
// memory profiles to cpu.prof and mem.prof:
//
//     go test -cpuprofile cpu.prof -memprofile mem.prof -bench .
//
// To add equivalent profiling support to a standalone program, add
// code like the following to your main function:
//
//    var cpuprofile = flag.String("cpuprofile", "", "write cpu profile `file`")
//    var memprofile = flag.String("memprofile", "", "write memory profile to `file`")
//
//    func main() {
//        flag.Parse()
//        if *cpuprofile != "" {
//            f, err := os.Create(*cpuprofile)
//            if err != nil {
//                log.Fatal("could not create CPU profile: ", err)
//            }
//            if err := pprof.StartCPUProfile(f); err != nil {
//                log.Fatal("could not start CPU profile: ", err)
//            }
//            defer pprof.StopCPUProfile()
//        }
//        ...
//        if *memprofile != "" {
//            f, err := os.Create(*memprofile)
//            if err != nil {
//                log.Fatal("could not create memory profile: ", err)
//            }
//            runtime.GC() // get up-to-date statistics
//            if err := pprof.WriteHeapProfile(f); err != nil {
//                log.Fatal("could not write memory profile: ", err)
//            }
//            f.Close()
//        }
//    }
//
// There is also a standard HTTP interface to profiling data. Adding
// the following line will install handlers under the /debug/pprof/
// URL to download live profiles:
//
//    import _ "net/http/pprof"
//
// See the net/http/pprof package for more details.
//
// Profiles can then be visualized with the pprof tool:
//
//    go tool pprof cpu.prof
//
// There are many commands available from the pprof command line.
// Commonly used commands include "top", which prints a summary of the
// top program hot-spots, and "web", which opens an interactive graph
// of hot-spots and their call graphs. Use "help" for information on
// all pprof commands.
//
// For more information about pprof, see
// https://github.com/google/pprof/blob/master/doc/pprof.md.
package pprof

import (
	"bufio"
	"bytes"
	"fmt"
	"internal/pprof/profile"
	"io"
	"runtime"
	"runtime/pprof/internal/protopprof"
	"sort"
	"strings"
	"sync"
	"text/tabwriter"
	"time"
)

// BUG(rsc): Profiles are only as good as the kernel support used to generate them.
// See https://golang.org/issue/13841 for details about known problems.

// A Profile is a collection of stack traces showing the call sequences
// that led to instances of a particular event, such as allocation.
// Packages can create and maintain their own profiles; the most common
// use is for tracking resources that must be explicitly closed, such as files
// or network connections.
//
// A Profile's methods can be called from multiple goroutines simultaneously.
//
// Each Profile has a unique name. A few profiles are predefined:
//
//	goroutine    - stack traces of all current goroutines
//	heap         - a sampling of all heap allocations
//	threadcreate - stack traces that led to the creation of new OS threads
//	block        - stack traces that led to blocking on synchronization primitives
//	mutex        - stack traces of holders of contended mutexes
//
// These predefined profiles maintain themselves and panic on an explicit
// Add or Remove method call.
//
// The heap profile reports statistics as of the most recently completed
// garbage collection; it elides more recent allocation to avoid skewing
// the profile away from live data and toward garbage.
// If there has been no garbage collection at all, the heap profile reports
// all known allocations. This exception helps mainly in programs running
// without garbage collection enabled, usually for debugging purposes.
//
// The CPU profile is not available as a Profile. It has a special API,
// the StartCPUProfile and StopCPUProfile functions, because it streams
// output to a writer during profiling.
//
type Profile struct {
	name  string
	mu    sync.Mutex
	m     map[interface{}][]uintptr
	count func() int
	write func(io.Writer, int) error
}

// profiles records all registered profiles.
var profiles struct {
	mu sync.Mutex
	m  map[string]*Profile
}

var goroutineProfile = &Profile{
	name:  "goroutine",
	count: countGoroutine,
	write: writeGoroutine,
}

var threadcreateProfile = &Profile{
	name:  "threadcreate",
	count: countThreadCreate,
	write: writeThreadCreate,
}

var heapProfile = &Profile{
	name:  "heap",
	count: countHeap,
	write: writeHeap,
}

var blockProfile = &Profile{
	name:  "block",
	count: countBlock,
	write: writeBlock,
}

var mutexProfile = &Profile{
	name:  "mutex",
	count: countMutex,
	write: writeMutex,
}

func lockProfiles() {
	profiles.mu.Lock()
	if profiles.m == nil {
		// Initial built-in profiles.
		profiles.m = map[string]*Profile{
			"goroutine":    goroutineProfile,
			"threadcreate": threadcreateProfile,
			"heap":         heapProfile,
			"block":        blockProfile,
			"mutex":        mutexProfile,
		}
	}
}

func unlockProfiles() {
	profiles.mu.Unlock()
}

// NewProfile creates a new profile with the given name.
// If a profile with that name already exists, NewProfile panics.
// The convention is to use a 'import/path.' prefix to create
// separate name spaces for each package.
func NewProfile(name string) *Profile {
	lockProfiles()
	defer unlockProfiles()
	if name == "" {
		panic("pprof: NewProfile with empty name")
	}
	if profiles.m[name] != nil {
		panic("pprof: NewProfile name already in use: " + name)
	}
	p := &Profile{
		name: name,
		m:    map[interface{}][]uintptr{},
	}
	profiles.m[name] = p
	return p
}

// Lookup returns the profile with the given name, or nil if no such profile exists.
func Lookup(name string) *Profile {
	lockProfiles()
	defer unlockProfiles()
	return profiles.m[name]
}

// Profiles returns a slice of all the known profiles, sorted by name.
func Profiles() []*Profile {
	lockProfiles()
	defer unlockProfiles()

	all := make([]*Profile, 0, len(profiles.m))
	for _, p := range profiles.m {
		all = append(all, p)
	}

	sort.Slice(all, func(i, j int) bool { return all[i].name < all[j].name })
	return all
}

// Name returns this profile's name, which can be passed to Lookup to reobtain the profile.
func (p *Profile) Name() string {
	return p.name
}

// Count returns the number of execution stacks currently in the profile.
func (p *Profile) Count() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.count != nil {
		return p.count()
	}
	return len(p.m)
}

// Add adds the current execution stack to the profile, associated with value.
// Add stores value in an internal map, so value must be suitable for use as
// a map key and will not be garbage collected until the corresponding
// call to Remove. Add panics if the profile already contains a stack for value.
//
// The skip parameter has the same meaning as runtime.Caller's skip
// and controls where the stack trace begins. Passing skip=0 begins the
// trace in the function calling Add. For example, given this
// execution stack:
//
//	Add
//	called from rpc.NewClient
//	called from mypkg.Run
//	called from main.main
//
// Passing skip=0 begins the stack trace at the call to Add inside rpc.NewClient.
// Passing skip=1 begins the stack trace at the call to NewClient inside mypkg.Run.
//
func (p *Profile) Add(value interface{}, skip int) {
	if p.name == "" {
		panic("pprof: use of uninitialized Profile")
	}
	if p.write != nil {
		panic("pprof: Add called on built-in Profile " + p.name)
	}

	stk := make([]uintptr, 32)
	n := runtime.Callers(skip+1, stk[:])

	p.mu.Lock()
	defer p.mu.Unlock()
	if p.m[value] != nil {
		panic("pprof: Profile.Add of duplicate value")
	}
	p.m[value] = stk[:n]
}

// Remove removes the execution stack associated with value from the profile.
// It is a no-op if the value is not in the profile.
func (p *Profile) Remove(value interface{}) {
	p.mu.Lock()
	defer p.mu.Unlock()
	delete(p.m, value)
}

// WriteTo writes a pprof-formatted snapshot of the profile to w.
// If a write to w returns an error, WriteTo returns that error.
// Otherwise, WriteTo returns nil.
//
// The debug parameter enables additional output.
// Passing debug=0 prints only the hexadecimal addresses that pprof needs.
// Passing debug=1 adds comments translating addresses to function names
// and line numbers, so that a programmer can read the profile without tools.
//
// The predefined profiles may assign meaning to other debug values;
// for example, when printing the "goroutine" profile, debug=2 means to
// print the goroutine stacks in the same form that a Go program uses
// when dying due to an unrecovered panic.
func (p *Profile) WriteTo(w io.Writer, debug int) error {
	if p.name == "" {
		panic("pprof: use of zero Profile")
	}
	if p.write != nil {
		return p.write(w, debug)
	}

	// Obtain consistent snapshot under lock; then process without lock.
	all := make([][]uintptr, 0, len(p.m))
	p.mu.Lock()
	for _, stk := range p.m {
		all = append(all, stk)
	}
	p.mu.Unlock()

	// Map order is non-deterministic; make output deterministic.
	sort.Sort(stackProfile(all))

	return printCountProfile(w, debug, p.name, stackProfile(all))
}

type stackProfile [][]uintptr

func (x stackProfile) Len() int              { return len(x) }
func (x stackProfile) Stack(i int) []uintptr { return x[i] }
func (x stackProfile) Swap(i, j int)         { x[i], x[j] = x[j], x[i] }
func (x stackProfile) Less(i, j int) bool {
	t, u := x[i], x[j]
	for k := 0; k < len(t) && k < len(u); k++ {
		if t[k] != u[k] {
			return t[k] < u[k]
		}
	}
	return len(t) < len(u)
}

// A countProfile is a set of stack traces to be printed as counts
// grouped by stack trace. There are multiple implementations:
// all that matters is that we can find out how many traces there are
// and obtain each trace in turn.
type countProfile interface {
	Len() int
	Stack(i int) []uintptr
}

// printCountProfile prints a countProfile at the specified debug level.
// The profile will be in compressed proto format unless debug is nonzero.
func printCountProfile(w io.Writer, debug int, name string, p countProfile) error {
	// Build count of each stack.
	var buf bytes.Buffer
	key := func(stk []uintptr) string {
		buf.Reset()
		fmt.Fprintf(&buf, "@")
		for _, pc := range stk {
			fmt.Fprintf(&buf, " %#x", pc)
		}
		return buf.String()
	}
	count := map[string]int{}
	index := map[string]int{}
	var keys []string
	n := p.Len()
	for i := 0; i < n; i++ {
		k := key(p.Stack(i))
		if count[k] == 0 {
			index[k] = i
			keys = append(keys, k)
		}
		count[k]++
	}

	sort.Sort(&keysByCount{keys, count})

	if debug > 0 {
		// Print debug profile in legacy format
		tw := tabwriter.NewWriter(w, 1, 8, 1, '\t', 0)
		fmt.Fprintf(tw, "%s profile: total %d\n", name, p.Len())
		for _, k := range keys {
			fmt.Fprintf(tw, "%d %s\n", count[k], k)
			printStackRecord(tw, p.Stack(index[k]), false)
		}
		return tw.Flush()
	}

	// Output profile in protobuf form.
	prof := &profile.Profile{
		PeriodType: &profile.ValueType{Type: name, Unit: "count"},
		Period:     1,
		Sample:     make([]*profile.Sample, 0, len(keys)),
		SampleType: []*profile.ValueType{{Type: name, Unit: "count"}},
	}
	locMap := make(map[uintptr]*profile.Location)
	for _, k := range keys {
		stk := p.Stack(index[k])
		c := count[k]
		locs := make([]*profile.Location, len(stk))
		for i, addr := range stk {
			loc := locMap[addr]
			if loc == nil {
				loc = &profile.Location{
					ID:      uint64(len(locMap) + 1),
					Address: uint64(addr - 1),
				}
				prof.Location = append(prof.Location, loc)
				locMap[addr] = loc
			}
			locs[i] = loc
		}
		prof.Sample = append(prof.Sample, &profile.Sample{
			Location: locs,
			Value:    []int64{int64(c)},
		})
	}
	return prof.Write(w)
}

// keysByCount sorts keys with higher counts first, breaking ties by key string order.
type keysByCount struct {
	keys  []string
	count map[string]int
}

func (x *keysByCount) Len() int      { return len(x.keys) }
func (x *keysByCount) Swap(i, j int) { x.keys[i], x.keys[j] = x.keys[j], x.keys[i] }
func (x *keysByCount) Less(i, j int) bool {
	ki, kj := x.keys[i], x.keys[j]
	ci, cj := x.count[ki], x.count[kj]
	if ci != cj {
		return ci > cj
	}
	return ki < kj
}

// printStackRecord prints the function + source line information
// for a single stack trace.
func printStackRecord(w io.Writer, stk []uintptr, allFrames bool) {
	show := allFrames
	frames := runtime.CallersFrames(stk)
	for {
		frame, more := frames.Next()
		name := frame.Function
		if name == "" {
			show = true
			fmt.Fprintf(w, "#\t%#x\n", frame.PC)
		} else if name != "runtime.goexit" && (show || !strings.HasPrefix(name, "runtime.")) {
			// Hide runtime.goexit and any runtime functions at the beginning.
			// This is useful mainly for allocation traces.
			show = true
			fmt.Fprintf(w, "#\t%#x\t%s+%#x\t%s:%d\n", frame.PC, name, frame.PC-frame.Entry, frame.File, frame.Line)
		}
		if !more {
			break
		}
	}
	if !show {
		// We didn't print anything; do it again,
		// and this time include runtime functions.
		printStackRecord(w, stk, true)
		return
	}
	fmt.Fprintf(w, "\n")
}

// Interface to system profiles.

// WriteHeapProfile is shorthand for Lookup("heap").WriteTo(w, 0).
// It is preserved for backwards compatibility.
func WriteHeapProfile(w io.Writer) error {
	return writeHeap(w, 0)
}

// countHeap returns the number of records in the heap profile.
func countHeap() int {
	n, _ := runtime.MemProfile(nil, true)
	return n
}

// writeHeap writes the current runtime heap profile to w.
func writeHeap(w io.Writer, debug int) error {
	// Find out how many records there are (MemProfile(nil, true)),
	// allocate that many records, and get the data.
	// There's a race—more records might be added between
	// the two calls—so allocate a few extra records for safety
	// and also try again if we're very unlucky.
	// The loop should only execute one iteration in the common case.
	var p []runtime.MemProfileRecord
	n, ok := runtime.MemProfile(nil, true)
	for {
		// Allocate room for a slightly bigger profile,
		// in case a few more entries have been added
		// since the call to MemProfile.
		p = make([]runtime.MemProfileRecord, n+50)
		n, ok = runtime.MemProfile(p, true)
		if ok {
			p = p[0:n]
			break
		}
		// Profile grew; try again.
	}

	if debug == 0 {
		pp := protopprof.EncodeMemProfile(p, int64(runtime.MemProfileRate), time.Now())
		return pp.Write(w)
	}

	sort.Slice(p, func(i, j int) bool { return p[i].InUseBytes() > p[j].InUseBytes() })

	b := bufio.NewWriter(w)
	tw := tabwriter.NewWriter(b, 1, 8, 1, '\t', 0)
	w = tw

	var total runtime.MemProfileRecord
	for i := range p {
		r := &p[i]
		total.AllocBytes += r.AllocBytes
		total.AllocObjects += r.AllocObjects
		total.FreeBytes += r.FreeBytes
		total.FreeObjects += r.FreeObjects
	}

	// Technically the rate is MemProfileRate not 2*MemProfileRate,
	// but early versions of the C++ heap profiler reported 2*MemProfileRate,
	// so that's what pprof has come to expect.
	fmt.Fprintf(w, "heap profile: %d: %d [%d: %d] @ heap/%d\n",
		total.InUseObjects(), total.InUseBytes(),
		total.AllocObjects, total.AllocBytes,
		2*runtime.MemProfileRate)

	for i := range p {
		r := &p[i]
		fmt.Fprintf(w, "%d: %d [%d: %d] @",
			r.InUseObjects(), r.InUseBytes(),
			r.AllocObjects, r.AllocBytes)
		for _, pc := range r.Stack() {
			fmt.Fprintf(w, " %#x", pc)
		}
		fmt.Fprintf(w, "\n")
		printStackRecord(w, r.Stack(), false)
	}

	// Print memstats information too.
	// Pprof will ignore, but useful for people
	s := new(runtime.MemStats)
	runtime.ReadMemStats(s)
	fmt.Fprintf(w, "\n# runtime.MemStats\n")
	fmt.Fprintf(w, "# Alloc = %d\n", s.Alloc)
	fmt.Fprintf(w, "# TotalAlloc = %d\n", s.TotalAlloc)
	fmt.Fprintf(w, "# Sys = %d\n", s.Sys)
	fmt.Fprintf(w, "# Lookups = %d\n", s.Lookups)
	fmt.Fprintf(w, "# Mallocs = %d\n", s.Mallocs)
	fmt.Fprintf(w, "# Frees = %d\n", s.Frees)

	fmt.Fprintf(w, "# HeapAlloc = %d\n", s.HeapAlloc)
	fmt.Fprintf(w, "# HeapSys = %d\n", s.HeapSys)
	fmt.Fprintf(w, "# HeapIdle = %d\n", s.HeapIdle)
	fmt.Fprintf(w, "# HeapInuse = %d\n", s.HeapInuse)
	fmt.Fprintf(w, "# HeapReleased = %d\n", s.HeapReleased)
	fmt.Fprintf(w, "# HeapObjects = %d\n", s.HeapObjects)

	fmt.Fprintf(w, "# Stack = %d / %d\n", s.StackInuse, s.StackSys)
	fmt.Fprintf(w, "# MSpan = %d / %d\n", s.MSpanInuse, s.MSpanSys)
	fmt.Fprintf(w, "# MCache = %d / %d\n", s.MCacheInuse, s.MCacheSys)
	fmt.Fprintf(w, "# BuckHashSys = %d\n", s.BuckHashSys)
	fmt.Fprintf(w, "# GCSys = %d\n", s.GCSys)
	fmt.Fprintf(w, "# OtherSys = %d\n", s.OtherSys)

	fmt.Fprintf(w, "# NextGC = %d\n", s.NextGC)
	fmt.Fprintf(w, "# PauseNs = %d\n", s.PauseNs)
	fmt.Fprintf(w, "# NumGC = %d\n", s.NumGC)
	fmt.Fprintf(w, "# DebugGC = %v\n", s.DebugGC)

	tw.Flush()
	return b.Flush()
}

// countThreadCreate returns the size of the current ThreadCreateProfile.
func countThreadCreate() int {
	n, _ := runtime.ThreadCreateProfile(nil)
	return n
}

// writeThreadCreate writes the current runtime ThreadCreateProfile to w.
func writeThreadCreate(w io.Writer, debug int) error {
	return writeRuntimeProfile(w, debug, "threadcreate", runtime.ThreadCreateProfile)
}

// countGoroutine returns the number of goroutines.
func countGoroutine() int {
	return runtime.NumGoroutine()
}

// writeGoroutine writes the current runtime GoroutineProfile to w.
func writeGoroutine(w io.Writer, debug int) error {
	if debug >= 2 {
		return writeGoroutineStacks(w)
	}
	return writeRuntimeProfile(w, debug, "goroutine", runtime.GoroutineProfile)
}

func writeGoroutineStacks(w io.Writer) error {
	// We don't know how big the buffer needs to be to collect
	// all the goroutines. Start with 1 MB and try a few times, doubling each time.
	// Give up and use a truncated trace if 64 MB is not enough.
	buf := make([]byte, 1<<20)
	for i := 0; ; i++ {
		n := runtime.Stack(buf, true)
		if n < len(buf) {
			buf = buf[:n]
			break
		}
		if len(buf) >= 64<<20 {
			// Filled 64 MB - stop there.
			break
		}
		buf = make([]byte, 2*len(buf))
	}
	_, err := w.Write(buf)
	return err
}

func writeRuntimeProfile(w io.Writer, debug int, name string, fetch func([]runtime.StackRecord) (int, bool)) error {
	// Find out how many records there are (fetch(nil)),
	// allocate that many records, and get the data.
	// There's a race—more records might be added between
	// the two calls—so allocate a few extra records for safety
	// and also try again if we're very unlucky.
	// The loop should only execute one iteration in the common case.
	var p []runtime.StackRecord
	n, ok := fetch(nil)
	for {
		// Allocate room for a slightly bigger profile,
		// in case a few more entries have been added
		// since the call to ThreadProfile.
		p = make([]runtime.StackRecord, n+10)
		n, ok = fetch(p)
		if ok {
			p = p[0:n]
			break
		}
		// Profile grew; try again.
	}

	return printCountProfile(w, debug, name, runtimeProfile(p))
}

type runtimeProfile []runtime.StackRecord

func (p runtimeProfile) Len() int              { return len(p) }
func (p runtimeProfile) Stack(i int) []uintptr { return p[i].Stack() }

var cpu struct {
	sync.Mutex
	profiling bool
	done      chan bool
}

// StartCPUProfile enables CPU profiling for the current process.
// While profiling, the profile will be buffered and written to w.
// StartCPUProfile returns an error if profiling is already enabled.
//
// On Unix-like systems, StartCPUProfile does not work by default for
// Go code built with -buildmode=c-archive or -buildmode=c-shared.
// StartCPUProfile relies on the SIGPROF signal, but that signal will
// be delivered to the main program's SIGPROF signal handler (if any)
// not to the one used by Go. To make it work, call os/signal.Notify
// for syscall.SIGPROF, but note that doing so may break any profiling
// being done by the main program.
func StartCPUProfile(w io.Writer) error {
	// The runtime routines allow a variable profiling rate,
	// but in practice operating systems cannot trigger signals
	// at more than about 500 Hz, and our processing of the
	// signal is not cheap (mostly getting the stack trace).
	// 100 Hz is a reasonable choice: it is frequent enough to
	// produce useful data, rare enough not to bog down the
	// system, and a nice round number to make it easy to
	// convert sample counts to seconds. Instead of requiring
	// each client to specify the frequency, we hard code it.
	const hz = 100

	cpu.Lock()
	defer cpu.Unlock()
	if cpu.done == nil {
		cpu.done = make(chan bool)
	}
	// Double-check.
	if cpu.profiling {
		return fmt.Errorf("cpu profiling already in use")
	}
	cpu.profiling = true
	runtime.SetCPUProfileRate(hz)
	go profileWriter(w)
	return nil
}

func profileWriter(w io.Writer) {
	startTime := time.Now()
	// This will buffer the entire profile into buf and then
	// translate it into a profile.Profile structure. This will
	// create two copies of all the data in the profile in memory.
	// TODO(matloob): Convert each chunk of the proto output and
	// stream it out instead of converting the entire profile.
	var buf bytes.Buffer
	for {
		data := runtime.CPUProfile()
		if data == nil {
			break
		}
		buf.Write(data)
	}

	profile, err := protopprof.TranslateCPUProfile(buf.Bytes(), startTime)
	if err != nil {
		// The runtime should never produce an invalid or truncated profile.
		// It drops records that can't fit into its log buffers.
		panic(fmt.Errorf("could not translate binary profile to proto format: %v", err))
	}

	profile.Write(w)
	cpu.done <- true
}

// StopCPUProfile stops the current CPU profile, if any.
// StopCPUProfile only returns after all the writes for the
// profile have completed.
func StopCPUProfile() {
	cpu.Lock()
	defer cpu.Unlock()

	if !cpu.profiling {
		return
	}
	cpu.profiling = false
	runtime.SetCPUProfileRate(0)
	<-cpu.done
}

// countBlock returns the number of records in the blocking profile.
func countBlock() int {
	n, _ := runtime.BlockProfile(nil)
	return n
}

// countMutex returns the number of records in the mutex profile.
func countMutex() int {
	n, _ := runtime.MutexProfile(nil)
	return n
}

// writeBlock writes the current blocking profile to w.
func writeBlock(w io.Writer, debug int) error {
	var p []runtime.BlockProfileRecord
	n, ok := runtime.BlockProfile(nil)
	for {
		p = make([]runtime.BlockProfileRecord, n+50)
		n, ok = runtime.BlockProfile(p)
		if ok {
			p = p[:n]
			break
		}
	}

	sort.Slice(p, func(i, j int) bool { return p[i].Cycles > p[j].Cycles })

	b := bufio.NewWriter(w)
	var tw *tabwriter.Writer
	w = b
	if debug > 0 {
		tw = tabwriter.NewWriter(w, 1, 8, 1, '\t', 0)
		w = tw
	}

	fmt.Fprintf(w, "--- contention:\n")
	fmt.Fprintf(w, "cycles/second=%v\n", runtime_cyclesPerSecond())
	for i := range p {
		r := &p[i]
		fmt.Fprintf(w, "%v %v @", r.Cycles, r.Count)
		for _, pc := range r.Stack() {
			fmt.Fprintf(w, " %#x", pc)
		}
		fmt.Fprint(w, "\n")
		if debug > 0 {
			printStackRecord(w, r.Stack(), true)
		}
	}

	if tw != nil {
		tw.Flush()
	}
	return b.Flush()
}

// writeMutex writes the current mutex profile to w.
func writeMutex(w io.Writer, debug int) error {
	// TODO(pjw): too much common code with writeBlock. FIX!
	var p []runtime.BlockProfileRecord
	n, ok := runtime.MutexProfile(nil)
	for {
		p = make([]runtime.BlockProfileRecord, n+50)
		n, ok = runtime.MutexProfile(p)
		if ok {
			p = p[:n]
			break
		}
	}

	sort.Slice(p, func(i, j int) bool { return p[i].Cycles > p[j].Cycles })

	b := bufio.NewWriter(w)
	var tw *tabwriter.Writer
	w = b
	if debug > 0 {
		tw = tabwriter.NewWriter(w, 1, 8, 1, '\t', 0)
		w = tw
	}

	fmt.Fprintf(w, "--- mutex:\n")
	fmt.Fprintf(w, "cycles/second=%v\n", runtime_cyclesPerSecond())
	fmt.Fprintf(w, "sampling period=%d\n", runtime.SetMutexProfileFraction(-1))
	for i := range p {
		r := &p[i]
		fmt.Fprintf(w, "%v %v @", r.Cycles, r.Count)
		for _, pc := range r.Stack() {
			fmt.Fprintf(w, " %#x", pc)
		}
		fmt.Fprint(w, "\n")
		if debug > 0 {
			printStackRecord(w, r.Stack(), true)
		}
	}

	if tw != nil {
		tw.Flush()
	}
	return b.Flush()
}

func runtime_cyclesPerSecond() int64

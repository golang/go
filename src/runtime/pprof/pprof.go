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
	"io"
	"math"
	"runtime"
	"sort"
	"strings"
	"sync"
	"text/tabwriter"
	"time"

	"runtime/pprof/internal/profile"
	"runtime/pprof/internal/protopprof"
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

// WriteTo writes a protobuf-formatted snapshot of the profile to w.
// If a write to w returns an error, WriteTo returns that error.
// Otherwise, WriteTo returns nil.
//
// The debug parameter enables adding names to locations.
// Passing debug=0 prints bare locations.
// Passing debug=1 adds translating addresses to function names
// and line numbers.
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

// Build count of stack.
func makeKey(stk []uintptr) string {
	var buf bytes.Buffer
	fmt.Fprintf(&buf, "@")
	for _, pc := range stk {
		fmt.Fprintf(&buf, " %#x", pc)
	}
	return buf.String()
}

// printCountProfile prints a countProfile at the specified debug level.
func printCountProfile(w io.Writer, debug int, name string, p countProfile) error {
	prof := &profile.Profile{
		PeriodType: &profile.ValueType{Type: name, Unit: "count"},
		Period:     1,
		SampleType: []*profile.ValueType{{Type: name, Unit: "count"}},
	}
	locations := make(map[uint64]*profile.Location)

	count := map[string]int{}
	index := map[string]int{}
	var keys []string
	n := p.Len()
	for i := 0; i < n; i++ {
		k := makeKey(p.Stack(i))
		if count[k] == 0 {
			index[k] = i
			keys = append(keys, k)
		}
		count[k]++
	}

	sort.Sort(&keysByCount{keys, count})

	// Print stacks, listing count on first occurrence of a unique stack.
	for _, k := range keys {
		stk := p.Stack(index[k])
		if c := count[k]; c != 0 {
			locs := make([]*profile.Location, 0, len(stk))
			for _, addr := range stk {
				addr := uint64(addr)
				// Adjust all frames by -1 to land on the call instruction.
				addr--
				loc := locations[addr]
				if loc == nil {
					loc = &profile.Location{
						Address: addr,
					}
					locations[addr] = loc
					prof.Location = append(prof.Location, loc)
				}
				locs = append(locs, loc)
			}
			prof.Sample = append(prof.Sample, &profile.Sample{
				Location: locs,
				Value:    []int64{int64(c)},
			})
			delete(count, k)
		}
	}

	prof.RemapAll()
	protopprof.Symbolize(prof)
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
		p = make([]runtime.MemProfileRecord, n+50)
		n, ok = runtime.MemProfile(p, true)
		if ok {
			p = p[0:n]
			break
		}
	}

	sort.Slice(p, func(i, j int) bool { return p[i].InUseBytes() > p[j].InUseBytes() })

	var total runtime.MemProfileRecord
	for i := range p {
		r := &p[i]
		total.AllocBytes += r.AllocBytes
		total.AllocObjects += r.AllocObjects
		total.FreeBytes += r.FreeBytes
		total.FreeObjects += r.FreeObjects
	}

	prof := &profile.Profile{
		PeriodType: &profile.ValueType{Type: "space", Unit: "bytes"},
		SampleType: []*profile.ValueType{
			{Type: "alloc_objects", Unit: "count"},
			{Type: "alloc_space", Unit: "bytes"},
			{Type: "inuse_objects", Unit: "count"},
			{Type: "inuse_space", Unit: "bytes"},
		},
		Period: int64(runtime.MemProfileRate),
	}

	locs := make(map[uint64]*(profile.Location))
	for i := range p {
		var v1, v2, v3, v4, blocksize int64
		r := &p[i]
		v1, v2 = int64(r.InUseObjects()), int64(r.InUseBytes())
		v3, v4 = int64(r.AllocObjects), int64(r.AllocBytes)
		if (v1 == 0 && v2 != 0) || (v3 == 0 && v4 != 0) {
			return fmt.Errorf("error writing memory profile: inuse object count was 0 but inuse bytes was %d", v2)
		} else {
			if v1 != 0 {
				blocksize = v2 / v1
				v1, v2 = scaleHeapSample(v1, v2, prof.Period)
			}
			if v3 != 0 {
				v3, v4 = scaleHeapSample(v3, v4, prof.Period)
			}
		}
		value := []int64{v1, v2, v3, v4}
		var sloc []*profile.Location
		for _, pc := range r.Stack() {
			addr := uint64(pc)
			addr--
			loc := locs[addr]
			if locs[addr] == nil {
				loc = &(profile.Location{
					Address: addr,
				})
				prof.Location = append(prof.Location, loc)
				locs[addr] = loc
			}
			sloc = append(sloc, loc)
		}
		prof.Sample = append(prof.Sample, &profile.Sample{
			Value:    value,
			Location: sloc,
			NumLabel: map[string][]int64{"bytes": {blocksize}},
		})
	}
	prof.RemapAll()
	protopprof.Symbolize(prof)
	return prof.Write(w)
}

// scaleHeapSample adjusts the data to account for its
// probability of appearing in the collected data.
func scaleHeapSample(count, size, rate int64) (int64, int64) {
	if count == 0 || size == 0 {
		return 0, 0
	}

	if rate <= 1 {
		// if rate==1 all samples were collected so no adjustment is needed.
		// if rate<1 treat as unknown and skip scaling.
		return count, size
	}

	// heap profiles rely on a poisson process to determine
	// which samples to collect, based on the desired average collection
	// rate R. The probability of a sample of size S to appear in that
	// profile is 1-exp(-S/R).
	avgSize := float64(size) / float64(count)
	scale := 1 / (1 - math.Exp(-avgSize/float64(rate)))

	return int64(float64(count) * scale), int64(float64(size) * scale)
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
	return writeRuntimeProfile(w, debug, "goroutine", runtime.GoroutineProfile)
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
	startTime time.Time
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
	var buf bytes.Buffer
	for {
		data := runtime.CPUProfile()
		buf.Write(data)
		if data == nil {
			break
		}
	}
	p, err := protopprof.TranslateCPUProfile(buf.Bytes(), cpu.startTime)
	if err != nil {
		panic(err)
	}
	p.RemapAll()
	protopprof.CleanupDuplicateLocations(p)
	protopprof.Symbolize(p)
	p.Write(w)
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
		// Code by analogy with writeBlock func
		p = make([]runtime.BlockProfileRecord, n+50)
		n, ok = runtime.BlockProfile(p)
		if ok {
			p = p[:n]
			break
		}
	}

	sort.Slice(p, func(i, j int) bool { return p[i].Cycles > p[j].Cycles })

	prof := &profile.Profile{
		PeriodType: &profile.ValueType{Type: "contentions", Unit: "count"},
		Period:     1,
		SampleType: []*profile.ValueType{
			{Type: "contentions", Unit: "count"},
			{Type: "delay", Unit: "nanoseconds"},
		},
	}

	cpuHz := runtime_cyclesPerSecond()
	locs := make(map[uint64]*profile.Location)
	for i := range p {
		r := &p[i]
		var v1, v2 int64
		v1 = r.Cycles
		v2 = r.Count
		if prof.Period > 0 {
			if cpuHz > 0 {
				cpuGHz := float64(cpuHz) / 1e9
				v1 = int64(float64(v1) * float64(prof.Period) / cpuGHz)
			}
			v2 = v2 * prof.Period
		}

		value := []int64{v2, v1}
		var sloc []*profile.Location

		for _, pc := range r.Stack() {
			addr := uint64(pc)
			addr--
			loc := locs[addr]
			if locs[addr] == nil {
				loc = &profile.Location{
					Address: addr,
				}
				prof.Location = append(prof.Location, loc)
				locs[addr] = loc
			}
			sloc = append(sloc, loc)
		}
		prof.Sample = append(prof.Sample, &profile.Sample{
			Value:    value,
			Location: sloc,
		})
	}

	prof.RemapAll()
	protopprof.Symbolize(prof)
	return prof.Write(w)
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

func runtime_cyclesPerSecond() int64

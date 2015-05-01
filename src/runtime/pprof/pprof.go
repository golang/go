// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package pprof writes runtime profiling data in the format expected
// by the pprof visualization tool.
// For more information about pprof, see
// http://code.google.com/p/google-perftools/.
package pprof

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"runtime"
	"sort"
	"strings"
	"sync"
	"text/tabwriter"
)

// BUG(rsc): Profiles are incomplete and inaccurate on NetBSD and OS X.
// See http://golang.org/issue/6047 for details.

// A Profile is a collection of stack traces showing the call sequences
// that led to instances of a particular event, such as allocation.
// Packages can create and maintain their own profiles; the most common
// use is for tracking resources that must be explicitly closed, such as files
// or network connections.
//
// A Profile's methods can be called from multiple goroutines simultaneously.
//
// Each Profile has a unique name.  A few profiles are predefined:
//
//	goroutine    - stack traces of all current goroutines
//	heap         - a sampling of all heap allocations
//	threadcreate - stack traces that led to the creation of new OS threads
//	block        - stack traces that led to blocking on synchronization primitives
//
// These predefined profiles maintain themselves and panic on an explicit
// Add or Remove method call.
//
// The CPU profile is not available as a Profile.  It has a special API,
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

func lockProfiles() {
	profiles.mu.Lock()
	if profiles.m == nil {
		// Initial built-in profiles.
		profiles.m = map[string]*Profile{
			"goroutine":    goroutineProfile,
			"threadcreate": threadcreateProfile,
			"heap":         heapProfile,
			"block":        blockProfile,
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

	var all []*Profile
	for _, p := range profiles.m {
		all = append(all, p)
	}

	sort.Sort(byName(all))
	return all
}

type byName []*Profile

func (x byName) Len() int           { return len(x) }
func (x byName) Swap(i, j int)      { x[i], x[j] = x[j], x[i] }
func (x byName) Less(i, j int) bool { return x[i].name < x[j].name }

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
// call to Remove.  Add panics if the profile already contains a stack for value.
//
// The skip parameter has the same meaning as runtime.Caller's skip
// and controls where the stack trace begins.  Passing skip=0 begins the
// trace in the function calling Add.  For example, given this
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
	var all [][]uintptr
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
// grouped by stack trace.  There are multiple implementations:
// all that matters is that we can find out how many traces there are
// and obtain each trace in turn.
type countProfile interface {
	Len() int
	Stack(i int) []uintptr
}

// printCountProfile prints a countProfile at the specified debug level.
func printCountProfile(w io.Writer, debug int, name string, p countProfile) error {
	b := bufio.NewWriter(w)
	var tw *tabwriter.Writer
	w = b
	if debug > 0 {
		tw = tabwriter.NewWriter(w, 1, 8, 1, '\t', 0)
		w = tw
	}

	fmt.Fprintf(w, "%s profile: total %d\n", name, p.Len())

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
	m := map[string]int{}
	n := p.Len()
	for i := 0; i < n; i++ {
		m[key(p.Stack(i))]++
	}

	// Print stacks, listing count on first occurrence of a unique stack.
	for i := 0; i < n; i++ {
		stk := p.Stack(i)
		s := key(stk)
		if count := m[s]; count != 0 {
			fmt.Fprintf(w, "%d %s\n", count, s)
			if debug > 0 {
				printStackRecord(w, stk, false)
			}
			delete(m, s)
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
	wasPanic := false
	for i, pc := range stk {
		f := runtime.FuncForPC(pc)
		if f == nil {
			show = true
			fmt.Fprintf(w, "#\t%#x\n", pc)
			wasPanic = false
		} else {
			tracepc := pc
			// Back up to call instruction.
			if i > 0 && pc > f.Entry() && !wasPanic {
				if runtime.GOARCH == "386" || runtime.GOARCH == "amd64" {
					tracepc--
				} else {
					tracepc -= 4 // arm, etc
				}
			}
			file, line := f.FileLine(tracepc)
			name := f.Name()
			// Hide runtime.goexit and any runtime functions at the beginning.
			// This is useful mainly for allocation traces.
			wasPanic = name == "runtime.panic"
			if name == "runtime.goexit" || !show && strings.HasPrefix(name, "runtime.") {
				continue
			}
			show = true
			fmt.Fprintf(w, "#\t%#x\t%s+%#x\t%s:%d\n", pc, name, pc-f.Entry(), file, line)
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

type byInUseBytes []runtime.MemProfileRecord

func (x byInUseBytes) Len() int           { return len(x) }
func (x byInUseBytes) Swap(i, j int)      { x[i], x[j] = x[j], x[i] }
func (x byInUseBytes) Less(i, j int) bool { return x[i].InUseBytes() > x[j].InUseBytes() }

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

	sort.Sort(byInUseBytes(p))

	b := bufio.NewWriter(w)
	var tw *tabwriter.Writer
	w = b
	if debug > 0 {
		tw = tabwriter.NewWriter(w, 1, 8, 1, '\t', 0)
		w = tw
	}

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
		if debug > 0 {
			printStackRecord(w, r.Stack(), false)
		}
	}

	// Print memstats information too.
	// Pprof will ignore, but useful for people
	if debug > 0 {
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

		fmt.Fprintf(w, "# NextGC = %d\n", s.NextGC)
		fmt.Fprintf(w, "# PauseNs = %d\n", s.PauseNs)
		fmt.Fprintf(w, "# NumGC = %d\n", s.NumGC)
		fmt.Fprintf(w, "# EnableGC = %v\n", s.EnableGC)
		fmt.Fprintf(w, "# DebugGC = %v\n", s.DebugGC)
	}

	if tw != nil {
		tw.Flush()
	}
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
	// all the goroutines.  Start with 1 MB and try a few times, doubling each time.
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
func StartCPUProfile(w io.Writer) error {
	// The runtime routines allow a variable profiling rate,
	// but in practice operating systems cannot trigger signals
	// at more than about 500 Hz, and our processing of the
	// signal is not cheap (mostly getting the stack trace).
	// 100 Hz is a reasonable choice: it is frequent enough to
	// produce useful data, rare enough not to bog down the
	// system, and a nice round number to make it easy to
	// convert sample counts to seconds.  Instead of requiring
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
	for {
		data := runtime.CPUProfile()
		if data == nil {
			break
		}
		w.Write(data)
	}
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

// TODO(rsc): Decide if StartTrace belongs in this package.
// See golang.org/issue/9710.
// StartTrace enables tracing for the current process.
// While tracing, the trace will be buffered and written to w.
// StartTrace returns an error if profiling is tracing enabled.
func StartTrace(w io.Writer) error {
	if err := runtime.StartTrace(); err != nil {
		return err
	}
	go func() {
		for {
			data := runtime.ReadTrace()
			if data == nil {
				break
			}
			w.Write(data)
		}
	}()
	return nil
}

// StopTrace stops the current tracing, if any.
// StopTrace only returns after all the writes for the trace have completed.
func StopTrace() {
	runtime.StopTrace()
}

type byCycles []runtime.BlockProfileRecord

func (x byCycles) Len() int           { return len(x) }
func (x byCycles) Swap(i, j int)      { x[i], x[j] = x[j], x[i] }
func (x byCycles) Less(i, j int) bool { return x[i].Cycles > x[j].Cycles }

// countBlock returns the number of records in the blocking profile.
func countBlock() int {
	n, _ := runtime.BlockProfile(nil)
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

	sort.Sort(byCycles(p))

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

func runtime_cyclesPerSecond() int64

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
	The runtime package contains operations that interact with Go's runtime system,
	such as functions to control goroutines. It also includes the low-level type information
	used by the reflect package; see reflect's documentation for the programmable
	interface to the run-time type system.
*/
package runtime

// Gosched yields the processor, allowing other goroutines to run.  It does not
// suspend the current goroutine, so execution resumes automatically.
func Gosched()

// Goexit terminates the goroutine that calls it.  No other goroutine is affected.
func Goexit()

// Breakpoint() executes a breakpoint trap.
func Breakpoint()

// Caller reports file and line number information about function invocations on
// the calling goroutine's stack.  The argument skip is the number of stack frames to
// ascend, with 0 identifying the the caller of Caller.  The return values report the
// program counter, file name, and line number within the file of the corresponding
// call.  The boolean ok is false if it was not possible to recover the information.
func Caller(skip int) (pc uintptr, file string, line int, ok bool)

// Callers fills the slice pc with the program counters of function invocations
// on the calling goroutine's stack.  The argument skip is the number of stack frames
// to skip before recording in pc, with 0 starting at the caller of Caller.
// It returns the number of entries written to pc.
func Callers(skip int, pc []uintptr) int

// FuncForPC returns a *Func describing the function that contains the
// given program counter address, or else nil.
func FuncForPC(pc uintptr) *Func

// NOTE(rsc): Func must match struct Func in runtime.h

// Func records information about a function in the program,
// in particular  the mapping from program counters to source
// line numbers within that function.
type Func struct {
	name   string
	typ    string
	src    string
	pcln   []byte
	entry  uintptr
	pc0    uintptr
	ln0    int32
	frame  int32
	args   int32
	locals int32
}

// Name returns the name of the function.
func (f *Func) Name() string { return f.name }

// Entry returns the entry address of the function.
func (f *Func) Entry() uintptr { return f.entry }

// FileLine returns the file name and line number of the
// source code corresponding to the program counter pc.
// The result will not be accurate if pc is not a program
// counter within f.
func (f *Func) FileLine(pc uintptr) (file string, line int) {
	// NOTE(rsc): If you edit this function, also edit
	// symtab.c:/^funcline.
	const PcQuant = 1

	p := f.pcln
	pc1 := f.pc0
	line = int(f.ln0)
	file = f.src
	for i := 0; i < len(p) && pc1 <= pc; i++ {
		switch {
		case p[i] == 0:
			line += int(p[i+1]<<24) | int(p[i+2]<<16) | int(p[i+3]<<8) | int(p[i+4])
			i += 4
		case p[i] <= 64:
			line += int(p[i])
		case p[i] <= 128:
			line += int(p[i] - 64)
		default:
			line += PcQuant * int(p[i]-129)
		}
		pc += PcQuant
	}
	return
}

// mid returns the current os thread (m) id.
func mid() uint32

// LockOSThread wires the calling goroutine to its current operating system thread.
// Until the calling goroutine exits or calls UnlockOSThread, it will always
// execute in that thread, and no other goroutine can.
// LockOSThread cannot be used during init functions.
func LockOSThread()

// UnlockOSThread unwires the calling goroutine from its fixed operating system thread.
// If the calling goroutine has not called LockOSThread, UnlockOSThread is a no-op.
func UnlockOSThread()

// GOMAXPROCS sets the maximum number of CPUs that can be executing
// simultaneously.   This call will go away when the scheduler improves.
func GOMAXPROCS(n int)

// Cgocalls returns the number of cgo calls made by the current process.
func Cgocalls() int64

// Semacquire waits until *s > 0 and then atomically decrements it.
// It is intended as a simple sleep primitive for use by the synchronization
// library and should not be used directly.
func Semacquire(s *uint32)

// Semrelease atomically increments *s and notifies a waiting goroutine
// if one is blocked in Semacquire.
// It is intended as a simple wakeup primitive for use by the synchronization
// library and should not be used directly.
func Semrelease(s *uint32)

// Sigrecv returns a bitmask of signals that have arrived since the last call to Sigrecv.
// It blocks until at least one signal arrives.
func Sigrecv() uint32

// Signame returns a string describing the signal, or "" if the signal is unknown.
func Signame(sig int32) string

// Siginit enables receipt of signals via Sigrecv.  It should typically
// be called during initialization.
func Siginit()

type MemStatsType struct {
	// General statistics.
	// Not locked during update; approximate.
	Alloc      uint64 // bytes allocated and still in use
	TotalAlloc uint64 // bytes allocated (even if freed)
	Sys        uint64 // bytes obtained from system (should be sum of XxxSys below)
	Lookups    uint64 // number of pointer lookups
	Mallocs    uint64 // number of mallocs

	// Main allocation heap statistics.
	HeapAlloc uint64 // bytes allocated and still in use
	HeapSys   uint64 // bytes obtained from system
	HeapIdle  uint64 // bytes in idle spans
	HeapInuse uint64 // bytes in non-idle span

	// Low-level fixed-size structure allocator statistics.
	//	Inuse is bytes used now.
	//	Sys is bytes obtained from system.
	StackInuse  uint64 // bootstrap stacks
	StackSys    uint64
	MSpanInuse  uint64 // mspan structures
	MSpanSys    uint64
	MCacheInuse uint64 // mcache structures
	MCacheSys   uint64

	// Garbage collector statistics.
	NextGC   uint64
	PauseNs  uint64
	NumGC    uint32
	EnableGC bool
	DebugGC  bool

	// Per-size allocation statistics.
	// Not locked during update; approximate.
	BySize [67]struct {
		Size    uint32
		Mallocs uint64
		Frees   uint64
	}
}

// MemStats holds statistics about the memory system.
// The statistics are only approximate, as they are not interlocked on update.
var MemStats MemStatsType

// Alloc allocates a block of the given size.
// FOR TESTING AND DEBUGGING ONLY.
func Alloc(uintptr) *byte

// Free frees the block starting at the given pointer.
// FOR TESTING AND DEBUGGING ONLY.
func Free(*byte)

// Lookup returns the base and size of the block containing the given pointer.
// FOR TESTING AND DEBUGGING ONLY.
func Lookup(*byte) (*byte, uintptr)

// GC runs a garbage collection.
func GC()

// SetFinalizer sets the finalizer associated with x to f.
// When the garbage collector finds an unreachable block
// with an associated finalizer, it clears the association and creates
// a new goroutine running f(x).  Creating the new goroutine makes
// x reachable again, but now without an associated finalizer.
// Assuming that SetFinalizer is not called again, the next time
// the garbage collector sees that x is unreachable, it will free x.
//
// SetFinalizer(x, nil) clears any finalizer associated with f.
//
// The argument x must be a pointer to an object allocated by
// calling new or by taking the address of a composite literal.
// The argument f must be a function that takes a single argument
// of x's type and returns no arguments.  If either of these is not
// true, SetFinalizer aborts the program.
//
// Finalizers are run in dependency order: if A points at B, both have
// finalizers, and they are otherwise unreachable, only the finalizer
// for A runs; once A is freed, the finalizer for B can run.
// If a cyclic structure includes a block with a finalizer, that
// cycle is not guaranteed to be garbage collected and the finalizer
// is not guaranteed to run, because there is no ordering that
// respects the dependencies.
//
// The finalizer for x is scheduled to run at some arbitrary time after
// x becomes unreachable.
// There is no guarantee that finalizers will run before a program exits,
// so typically they are useful only for releasing non-memory resources
// associated with an object during a long-running program.
// For example, an os.File object could use a finalizer to close the
// associated operating system file descriptor when a program discards
// an os.File without calling Close, but it would be a mistake
// to depend on a finalizer to flush an in-memory I/O buffer such as a
// bufio.Writer, because the buffer would not be flushed at program exit.
//
// A single goroutine runs all finalizers for a program, sequentially.
// If a finalizer must run for a long time, it should do so by starting
// a new goroutine.
//
// TODO(rsc): make os.File use SetFinalizer
// TODO(rsc): allow f to have (ignored) return values
//
func SetFinalizer(x, f interface{})

func getgoroot() string

// GOROOT returns the root of the Go tree.
// It uses the GOROOT environment variable, if set,
// or else the root used during the Go build.
func GOROOT() string {
	s := getgoroot()
	if s != "" {
		return s
	}
	return defaultGoroot
}

// Version returns the Go tree's version string.
// It is either a sequence number or, when possible,
// a release tag like "release.2010-03-04".
// A trailing + indicates that the tree had local modifications
// at the time of the build.
func Version() string { return defaultVersion }

// MemProfileRate controls the fraction of memory allocations
// that are recorded and reported in the memory profile.
// The profiler aims to sample an average of
// one allocation per MemProfileRate bytes allocated.
//
// To include every allocated block in the profile, set MemProfileRate to 1.
// To turn off profiling entirely, set MemProfileRate to 0.
//
// The tools that process the memory profiles assume that the
// profile rate is constant across the lifetime of the program
// and equal to the current value.  Programs that change the
// memory profiling rate should do so just once, as early as
// possible in the execution of the program (for example,
// at the beginning of main).
var MemProfileRate int = 512 * 1024

// A MemProfileRecord describes the live objects allocated
// by a particular call sequence (stack trace).
type MemProfileRecord struct {
	AllocBytes, FreeBytes     int64       // number of bytes allocated, freed
	AllocObjects, FreeObjects int64       // number of objects allocated, freed
	Stack0                    [32]uintptr // stack trace for this record; ends at first 0 entry
}

// InUseBytes returns the number of bytes in use (AllocBytes - FreeBytes).
func (r *MemProfileRecord) InUseBytes() int64 { return r.AllocBytes - r.FreeBytes }

// InUseObjects returns the number of objects in use (AllocObjects - FreeObjects).
func (r *MemProfileRecord) InUseObjects() int64 {
	return r.AllocObjects - r.FreeObjects
}

// Stack returns the stack trace associated with the record,
// a prefix of r.Stack0.
func (r *MemProfileRecord) Stack() []uintptr {
	for i, v := range r.Stack0 {
		if v == 0 {
			return r.Stack0[0:i]
		}
	}
	return r.Stack0[0:]
}

// MemProfile returns n, the number of records in the current memory profile.
// If len(p) >= n, MemProfile copies the profile into p and returns n, true.
// If len(p) < n, MemProfile does not change p and returns n, false.
//
// If inuseZero is true, the profile includes allocation records
// where r.AllocBytes > 0 but r.AllocBytes == r.FreeBytes.
// These are sites where memory was allocated, but it has all
// been released back to the runtime.
func MemProfile(p []MemProfileRecord, inuseZero bool) (n int, ok bool)

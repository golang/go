// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.19

package counter

// The implementation of this package and tests are located in
// internal/counter, which can be shared with the upload package.
// TODO(hyangah): use of type aliases prevents nice documentation
// rendering in go doc or pkgsite. Fix this either by avoiding
// type aliasing or restructuring the internal/counter package.
import (
	"flag"

	"golang.org/x/telemetry/internal/counter"
)

// Inc increments the counter with the given name.
func Inc(name string) {
	New(name).Inc()
}

// Add adds n to the counter with the given name.
func Add(name string, n int64) {
	New(name).Add(n)
}

// New returns a counter with the given name.
// New can be called in global initializers and will be compiled down to
// linker-initialized data. That is, calling New to initialize a global
// has no cost at program startup.
func New(name string) *Counter {
	// Note: not calling DefaultFile.New in order to keep this
	// function something the compiler can inline and convert
	// into static data initializations, with no init-time footprint.
	// TODO(hyangah): is it trivial enough for the compiler to inline?
	return counter.New(name)
}

// A Counter is a single named event counter.
// A Counter is safe for use by multiple goroutines simultaneously.
//
// Counters should typically be created using New
// and stored as global variables, like:
//
//	package mypackage
//	var errorCount = counter.New("mypackage/errors")
//
// (The initialization of errorCount in this example is handled
// entirely by the compiler and linker; this line executes no code
// at program startup.)
//
// Then code can call Add to increment the counter
// each time the corresponding event is observed.
//
// Although it is possible to use New to create
// a Counter each time a particular event needs to be recorded,
// that usage fails to amortize the construction cost over
// multiple calls to Add, so it is more expensive and not recommended.
type Counter = counter.Counter

// a StackCounter is the in-memory knowledge about a stack counter.
// StackCounters are more expensive to use than regular Counters,
// requiring, at a minimum, a call to runtime.Callers.
type StackCounter = counter.StackCounter

// NewStack returns a new stack counter with the given name and depth.
func NewStack(name string, depth int) *StackCounter {
	return counter.NewStack(name, depth)
}

// Open prepares telemetry counters for recording to the file system.
//
// If the telemetry mode is "off", Open is a no-op. Otherwise, it opens the
// counter file on disk and starts to mmap telemetry counters to the file.
// Open also persists any counters already created in the current process.
//
// Programs using telemetry should call Open exactly once.
func Open() {
	counter.Open()
}

// CountFlags creates a counter for every flag that is set
// and increments the counter. The name of the counter is
// the concatenation of prefix and the flag name.
//
//	For instance, CountFlags("gopls:flag-", flag.CommandLine)
func CountFlags(prefix string, fs flag.FlagSet) {
	fs.Visit(func(f *flag.Flag) {
		New(prefix + f.Name).Inc()
	})
}

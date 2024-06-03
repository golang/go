// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package internal/counter implements the internals of the public counter package.
// In addition to the public API, this package also includes APIs to parse and
// manage the counter files, needed by the upload package.
package counter

import (
	"fmt"
	"os"
	"runtime"
	"strings"
	"sync/atomic"
)

// Note: not using internal/godebug, so that internal/godebug can use internal/counter.
var debugCounter = strings.Contains(os.Getenv("GODEBUG"), "countertrace=1")

func debugPrintf(format string, args ...interface{}) {
	if debugCounter {
		if len(format) == 0 || format[len(format)-1] != '\n' {
			format += "\n"
		}
		fmt.Fprintf(os.Stderr, "counter: "+format, args...)
	}
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
type Counter struct {
	name string
	file *file

	next  atomic.Pointer[Counter]
	state counterState
	ptr   counterPtr
}

func (c *Counter) Name() string {
	return c.name
}

type counterPtr struct {
	m     *mappedFile
	count *atomic.Uint64
}

type counterState struct {
	bits atomic.Uint64
}

func (s *counterState) load() counterStateBits {
	return counterStateBits(s.bits.Load())
}

func (s *counterState) update(old *counterStateBits, new counterStateBits) bool {
	if s.bits.CompareAndSwap(uint64(*old), uint64(new)) {
		*old = new
		return true
	}
	return false
}

type counterStateBits uint64

const (
	stateReaders    counterStateBits = 1<<30 - 1
	stateLocked     counterStateBits = stateReaders
	stateHavePtr    counterStateBits = 1 << 30
	stateExtraShift                  = 31
	stateExtra      counterStateBits = 1<<64 - 1<<stateExtraShift
)

func (b counterStateBits) readers() int  { return int(b & stateReaders) }
func (b counterStateBits) locked() bool  { return b&stateReaders == stateLocked }
func (b counterStateBits) havePtr() bool { return b&stateHavePtr != 0 }
func (b counterStateBits) extra() uint64 { return uint64(b&stateExtra) >> stateExtraShift }

func (b counterStateBits) incReader() counterStateBits    { return b + 1 }
func (b counterStateBits) decReader() counterStateBits    { return b - 1 }
func (b counterStateBits) setLocked() counterStateBits    { return b | stateLocked }
func (b counterStateBits) clearLocked() counterStateBits  { return b &^ stateLocked }
func (b counterStateBits) setHavePtr() counterStateBits   { return b | stateHavePtr }
func (b counterStateBits) clearHavePtr() counterStateBits { return b &^ stateHavePtr }
func (b counterStateBits) clearExtra() counterStateBits   { return b &^ stateExtra }
func (b counterStateBits) addExtra(n uint64) counterStateBits {
	const maxExtra = uint64(stateExtra) >> stateExtraShift // 0x1ffffffff
	x := b.extra()
	if x+n < x || x+n > maxExtra {
		x = maxExtra
	} else {
		x += n
	}
	return b.clearExtra() | counterStateBits(x)<<stateExtraShift
}

// New returns a counter with the given name.
// New can be called in global initializers and will be compiled down to
// linker-initialized data. That is, calling New to initialize a global
// has no cost at program startup.
func New(name string) *Counter {
	// Note: not calling defaultFile.New in order to keep this
	// function something the compiler can inline and convert
	// into static data initializations, with no init-time footprint.
	return &Counter{name: name, file: &defaultFile}
}

// Inc adds 1 to the counter.
func (c *Counter) Inc() {
	c.Add(1)
}

// Add adds n to the counter. n cannot be negative, as counts cannot decrease.
func (c *Counter) Add(n int64) {
	debugPrintf("Add %q += %d", c.name, n)

	if n < 0 {
		panic("Counter.Add negative")
	}
	if n == 0 {
		return
	}
	c.file.register(c)

	state := c.state.load()
	for ; ; state = c.state.load() {
		switch {
		case !state.locked() && state.havePtr():
			if !c.state.update(&state, state.incReader()) {
				continue
			}
			// Counter unlocked or counter shared; has an initialized count pointer; acquired shared lock.
			if c.ptr.count == nil {
				for !c.state.update(&state, state.addExtra(uint64(n))) {
					// keep trying - we already took the reader lock
					state = c.state.load()
				}
				debugPrintf("Add %q += %d: nil extra=%d\n", c.name, n, state.extra())
			} else {
				sum := c.add(uint64(n))
				debugPrintf("Add %q += %d: count=%d\n", c.name, n, sum)
			}
			c.releaseReader(state)
			return

		case state.locked():
			if !c.state.update(&state, state.addExtra(uint64(n))) {
				continue
			}
			debugPrintf("Add %q += %d: locked extra=%d\n", c.name, n, state.extra())
			return

		case !state.havePtr():
			if !c.state.update(&state, state.addExtra(uint64(n)).setLocked()) {
				continue
			}
			debugPrintf("Add %q += %d: noptr extra=%d\n", c.name, n, state.extra())
			c.releaseLock(state)
			return
		}
	}
}

func (c *Counter) releaseReader(state counterStateBits) {
	for ; ; state = c.state.load() {
		// If we are the last reader and havePtr was cleared
		// while this batch of readers was using c.ptr,
		// it's our job to update c.ptr by upgrading to a full lock
		// and letting releaseLock do the work.
		// Note: no new reader will attempt to add itself now that havePtr is clear,
		// so we are only racing against possible additions to extra.
		if state.readers() == 1 && !state.havePtr() {
			if !c.state.update(&state, state.setLocked()) {
				continue
			}
			debugPrintf("releaseReader %s: last reader, need ptr\n", c.name)
			c.releaseLock(state)
			return
		}

		// Release reader.
		if !c.state.update(&state, state.decReader()) {
			continue
		}
		debugPrintf("releaseReader %s: released (%d readers now)\n", c.name, state.readers())
		return
	}
}

func (c *Counter) releaseLock(state counterStateBits) {
	for ; ; state = c.state.load() {
		if !state.havePtr() {
			// Set havePtr before updating ptr,
			// to avoid race with the next clear of havePtr.
			if !c.state.update(&state, state.setHavePtr()) {
				continue
			}
			debugPrintf("releaseLock %s: reset havePtr (extra=%d)\n", c.name, state.extra())

			// Optimization: only bother loading a new pointer
			// if we have a value to add to it.
			c.ptr = counterPtr{nil, nil}
			if state.extra() != 0 {
				c.ptr = c.file.lookup(c.name)
				debugPrintf("releaseLock %s: ptr=%v\n", c.name, c.ptr)
			}
		}

		if extra := state.extra(); extra != 0 && c.ptr.count != nil {
			if !c.state.update(&state, state.clearExtra()) {
				continue
			}
			sum := c.add(extra)
			debugPrintf("releaseLock %s: flush extra=%d -> count=%d\n", c.name, extra, sum)
		}

		// Took care of refreshing ptr and flushing extra.
		// Now we can release the lock, unless of course
		// another goroutine cleared havePtr or added to extra,
		// in which case we go around again.
		if !c.state.update(&state, state.clearLocked()) {
			continue
		}
		debugPrintf("releaseLock %s: unlocked\n", c.name)
		return
	}
}

func (c *Counter) add(n uint64) uint64 {
	count := c.ptr.count
	for {
		old := count.Load()
		sum := old + n
		if sum < old {
			sum = ^uint64(0)
		}
		if count.CompareAndSwap(old, sum) {
			runtime.KeepAlive(c.ptr.m)
			return sum
		}
	}
}

func (c *Counter) invalidate() {
	for {
		state := c.state.load()
		if !state.havePtr() {
			debugPrintf("invalidate %s: no ptr\n", c.name)
			return
		}
		if c.state.update(&state, state.clearHavePtr()) {
			debugPrintf("invalidate %s: cleared havePtr\n", c.name)
			return
		}
	}
}

func (c *Counter) refresh() {
	for {
		state := c.state.load()
		if state.havePtr() || state.readers() > 0 || state.extra() == 0 {
			debugPrintf("refresh %s: havePtr=%v readers=%d extra=%d\n", c.name, state.havePtr(), state.readers(), state.extra())
			return
		}
		if c.state.update(&state, state.setLocked()) {
			debugPrintf("refresh %s: locked havePtr=%v readers=%d extra=%d\n", c.name, state.havePtr(), state.readers(), state.extra())
			c.releaseLock(state)
			return
		}
	}
}

// Read reads the given counter.
// This is the implementation of x/telemetry/counter/countertest.ReadCounter.
func Read(c *Counter) (uint64, error) {
	if c.file.current.Load() == nil {
		return c.state.load().extra(), nil
	}
	pf, err := readFile(c.file)
	if err != nil {
		return 0, err
	}
	v, ok := pf.Count[DecodeStack(c.Name())]
	if !ok {
		return v, fmt.Errorf("not found:%q", DecodeStack(c.Name()))
	}
	return v, nil
}

func readFile(f *file) (*File, error) {
	if f == nil {
		debugPrintf("No file")
		return nil, fmt.Errorf("counter is not initialized - was Open called?")
	}

	// Note: don't call f.rotate here as this will enqueue a follow-up rotation.
	_, cleanup := f.rotate1()
	cleanup()

	if f.err != nil {
		return nil, fmt.Errorf("failed to rotate mapped file - %v", f.err)
	}
	current := f.current.Load()
	if current == nil {
		return nil, fmt.Errorf("counter has no mapped file")
	}
	name := current.f.Name()
	data, err := os.ReadFile(name)
	if err != nil {
		return nil, fmt.Errorf("failed to read from file: %v", err)
	}
	pf, err := Parse(name, data)
	if err != nil {
		return nil, fmt.Errorf("failed to parse: %v", err)
	}
	return pf, nil
}

// ReadFile reads the counters and stack counters from the given file.
// This is the implementation of x/telemetry/counter/countertest.Read
func ReadFile(name string) (counters, stackCounters map[string]uint64, _ error) {
	// TODO: Document the format of the stackCounters names.

	data, err := os.ReadFile(name)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to read from file: %v", err)
	}
	pf, err := Parse(name, data)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to parse: %v", err)
	}
	counters = make(map[string]uint64)
	stackCounters = make(map[string]uint64)
	for k, v := range pf.Count {
		if IsStackCounter(k) {
			stackCounters[DecodeStack(k)] = v
		} else {
			counters[k] = v
		}
	}
	return counters, stackCounters, nil
}

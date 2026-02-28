// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package counter

import (
	"fmt"
	"runtime"
	"strings"
	"sync"
)

// On the disk, and upstream, stack counters look like sets of
// regular counters with names that include newlines.

// a StackCounter is the in-memory knowledge about a stack counter.
// StackCounters are more expensive to use than regular Counters,
// requiring, at a minimum, a call to runtime.Callers.
type StackCounter struct {
	name  string
	depth int
	file  *file

	mu sync.Mutex
	// as this is a detail of the implementation, it could be replaced
	// by a more efficient mechanism
	stacks []stack
}

type stack struct {
	pcs     []uintptr
	counter *Counter
}

func NewStack(name string, depth int) *StackCounter {
	return &StackCounter{name: name, depth: depth, file: &defaultFile}
}

// Inc increments a stack counter. It computes the caller's stack and
// looks up the corresponding counter. It then increments that counter,
// creating it if necessary.
func (c *StackCounter) Inc() {
	pcs := make([]uintptr, c.depth)
	n := runtime.Callers(2, pcs) // caller of Inc
	pcs = pcs[:n]

	c.mu.Lock()
	defer c.mu.Unlock()

	// Existing counter?
	var ctr *Counter
	for _, s := range c.stacks {
		if eq(s.pcs, pcs) {
			if s.counter != nil {
				ctr = s.counter
				break
			}
		}
	}

	if ctr == nil {
		// Create new counter.
		ctr = &Counter{
			name: EncodeStack(pcs, c.name),
			file: c.file,
		}
		c.stacks = append(c.stacks, stack{pcs: pcs, counter: ctr})
	}

	ctr.Inc()
}

// EncodeStack returns the name of the counter to
// use for the given stack of program counters.
// The name encodes the stack.
func EncodeStack(pcs []uintptr, prefix string) string {
	var locs []string
	lastImport := ""
	frs := runtime.CallersFrames(pcs)
	for {
		fr, more := frs.Next()
		// TODO(adonovan): this CutLast(".") operation isn't
		// appropriate for generic function symbols.
		path, fname := cutLastDot(fr.Function)
		if path == lastImport {
			path = `"` // (a ditto mark)
		} else {
			lastImport = path
		}
		var loc string
		if fr.Func != nil {
			// Use function-relative line numbering.
			// f:+2 means two lines into function f.
			// f:-1 should never happen, but be conservative.
			//
			// An inlined call is replaced by a NOP instruction
			// with the correct pclntab information.
			_, entryLine := fr.Func.FileLine(fr.Entry)
			loc = fmt.Sprintf("%s.%s:%+d,+0x%x", path, fname, fr.Line-entryLine, fr.PC-fr.Entry)
		} else {
			// The function is non-Go code or is fully inlined:
			// use absolute line number within enclosing file.
			//
			// For inlined calls, the PC and Entry values
			// both refer to the enclosing combined function.
			// For example, both these PCs are relative to "caller":
			//
			//   callee:=1,+0x12        ('=' means inlined)
			//   caller:+2,+0x34
			loc = fmt.Sprintf("%s.%s:=%d,+0x%x", path, fname, fr.Line, fr.PC-fr.Entry)
		}
		locs = append(locs, loc)
		if !more {
			break
		}
	}

	name := prefix + "\n" + strings.Join(locs, "\n")
	if len(name) > maxNameLen {
		const bad = "\ntruncated\n"
		name = name[:maxNameLen-len(bad)] + bad
	}
	return name
}

// DecodeStack expands the (compressed) stack encoded in the counter name.
func DecodeStack(ename string) string {
	if !strings.Contains(ename, "\n") {
		return ename // not a stack counter
	}
	lines := strings.Split(ename, "\n")
	var lastPath string // empty or ends with .
	for i, line := range lines {
		path, rest := cutLastDot(line)
		if len(path) == 0 {
			continue // unchanged
		}
		if len(path) == 1 && path[0] == '"' {
			lines[i] = lastPath + rest
		} else {
			lastPath = path + "."
			// line unchanged
		}
	}
	return strings.Join(lines, "\n") // trailing \n?
}

// input is <import path>.<function name>
// output is (import path, function name)
func cutLastDot(x string) (before, after string) {
	i := strings.LastIndex(x, ".")
	if i < 0 {
		return "", x
	}
	return x[:i], x[i+1:]
}

// Names reports all the counter names associated with a StackCounter.
func (c *StackCounter) Names() []string {
	c.mu.Lock()
	defer c.mu.Unlock()
	names := make([]string, len(c.stacks))
	for i, s := range c.stacks {
		names[i] = s.counter.Name()
	}
	return names
}

// Counters returns the known Counters for a StackCounter.
// There may be more in the count file.
func (c *StackCounter) Counters() []*Counter {
	c.mu.Lock()
	defer c.mu.Unlock()
	counters := make([]*Counter, len(c.stacks))
	for i, s := range c.stacks {
		counters[i] = s.counter
	}
	return counters
}

func eq(a, b []uintptr) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// ReadStack reads the given stack counter.
// This is the implementation of
// golang.org/x/telemetry/counter/countertest.ReadStackCounter.
func ReadStack(c *StackCounter) (map[string]uint64, error) {
	ret := map[string]uint64{}
	for _, ctr := range c.Counters() {
		v, err := Read(ctr)
		if err != nil {
			return nil, err
		}
		ret[DecodeStack(ctr.Name())] = v
	}
	return ret, nil
}

// IsStackCounter reports whether the counter name is for a stack counter.
func IsStackCounter(name string) bool {
	return strings.Contains(name, "\n")
}

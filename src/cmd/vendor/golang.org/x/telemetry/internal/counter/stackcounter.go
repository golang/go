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
	for _, s := range c.stacks {
		if eq(s.pcs, pcs) {
			if s.counter != nil {
				s.counter.Inc()
			}
			return
		}
	}
	// have to create the new counter's name, and the new counter itself
	locs := make([]string, 0, c.depth)
	lastImport := ""
	frs := runtime.CallersFrames(pcs)
	for i := 0; ; i++ {
		fr, more := frs.Next()
		pcline := fr.Line
		entryptr := fr.Entry
		var locline string
		path, fname := splitPath(fr.Function)
		if path == lastImport {
			path = "\""
		} else {
			lastImport = path
		}
		if fr.Func != nil {
			_, entryline := fr.Func.FileLine(entryptr)
			if pcline >= entryline {
				locline = fmt.Sprintf("%s.%s:%d", path, fname, pcline-entryline)
			} else {
				// unexpected
				locline = fmt.Sprintf("%s.%s:??%d", path, fname, pcline)
				lastImport = ""
			}
		} else {
			// might happen if the function is non-Go code or is fully inlined.
			locline = fmt.Sprintf("%s.%s:?%d", path, fname, pcline)
			lastImport = ""
		}
		locs = append(locs, locline)
		if !more {
			break
		}
	}

	name := c.name + "\n" + strings.Join(locs, "\n")
	if len(name) > maxNameLen {
		const bad = "\ntruncated\n"
		name = name[:maxNameLen-len(bad)] + bad
	}
	ctr := &Counter{name: name, file: c.file}
	c.stacks = append(c.stacks, stack{pcs: pcs, counter: ctr})
	ctr.Inc()
}

// input is <import path>.<function name>
// output is (import path, function name)
func splitPath(x string) (string, string) {
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
	pf, err := readFile(c.file)
	if err != nil {
		return nil, err
	}
	ret := map[string]uint64{}
	prefix := c.name + "\n"

	for k, v := range pf.Count {
		if strings.HasPrefix(k, prefix) {
			ret[k] = v
		}
	}
	return ret, nil
}

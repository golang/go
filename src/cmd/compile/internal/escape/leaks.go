// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package escape

import (
	"cmd/compile/internal/base"
	"math"
	"strings"
)

// A leaks represents a set of assignment flows from a parameter to
// the heap, mutator, callee, or to any of its function's (first
// numEscResults) result parameters.
type leaks [8]uint8

const (
	leakHeap = iota
	leakMutator
	leakCallee
	leakResult0
)

const numEscResults = len(leaks{}) - leakResult0

// Heap returns the minimum deref count of any assignment flow from l
// to the heap. If no such flows exist, Heap returns -1.
func (l leaks) Heap() int { return l.get(leakHeap) }

// Mutator returns the minimum deref count of any assignment flow from
// l to the pointer operand of an indirect assignment statement. If no
// such flows exist, Mutator returns -1.
func (l leaks) Mutator() int { return l.get(leakMutator) }

// Callee returns the minimum deref count of any assignment flow from
// l to the callee operand of call expression. If no such flows exist,
// Callee returns -1.
func (l leaks) Callee() int { return l.get(leakCallee) }

// Result returns the minimum deref count of any assignment flow from
// l to its function's i'th result parameter. If no such flows exist,
// Result returns -1.
func (l leaks) Result(i int) int { return l.get(leakResult0 + i) }

// AddHeap adds an assignment flow from l to the heap.
func (l *leaks) AddHeap(derefs int) { l.add(leakHeap, derefs) }

// AddMutator adds a flow from l to the mutator (i.e., a pointer
// operand of an indirect assignment statement).
func (l *leaks) AddMutator(derefs int) { l.add(leakMutator, derefs) }

// AddCallee adds an assignment flow from l to the callee operand of a
// call expression.
func (l *leaks) AddCallee(derefs int) { l.add(leakCallee, derefs) }

// AddResult adds an assignment flow from l to its function's i'th
// result parameter.
func (l *leaks) AddResult(i, derefs int) { l.add(leakResult0+i, derefs) }

func (l leaks) get(i int) int { return int(l[i]) - 1 }

func (l *leaks) add(i, derefs int) {
	if old := l.get(i); old < 0 || derefs < old {
		l.set(i, derefs)
	}
}

func (l *leaks) set(i, derefs int) {
	v := derefs + 1
	if v < 0 {
		base.Fatalf("invalid derefs count: %v", derefs)
	}
	if v > math.MaxUint8 {
		v = math.MaxUint8
	}

	l[i] = uint8(v)
}

// Optimize removes result flow paths that are equal in length or
// longer than the shortest heap flow path.
func (l *leaks) Optimize() {
	// If we have a path to the heap, then there's no use in
	// keeping equal or longer paths elsewhere.
	if x := l.Heap(); x >= 0 {
		for i := 1; i < len(*l); i++ {
			if l.get(i) >= x {
				l.set(i, -1)
			}
		}
	}
}

var leakTagCache = map[leaks]string{}

// Encode converts l into a binary string for export data.
func (l leaks) Encode() string {
	if l.Heap() == 0 {
		// Space optimization: empty string encodes more
		// efficiently in export data.
		return ""
	}
	if s, ok := leakTagCache[l]; ok {
		return s
	}

	n := len(l)
	for n > 0 && l[n-1] == 0 {
		n--
	}
	s := "esc:" + string(l[:n])
	leakTagCache[l] = s
	return s
}

// parseLeaks parses a binary string representing a leaks.
func parseLeaks(s string) leaks {
	var l leaks
	if !strings.HasPrefix(s, "esc:") {
		l.AddHeap(0)
		return l
	}
	copy(l[:], s[4:])
	return l
}

func ParseLeaks(s string) leaks {
	return parseLeaks(s)
}

// Any reports whether the value flows anywhere at all.
func (l leaks) Any() bool {
	// TODO: do mutator/callee matter?
	if l.Heap() >= 0 || l.Mutator() >= 0 || l.Callee() >= 0 {
		return true
	}
	for i := range numEscResults {
		if l.Result(i) >= 0 {
			return true
		}
	}
	return false
}

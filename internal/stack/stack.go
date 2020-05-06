// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package stack provides support for parsing standard goroutine stack traces.
package stack

import (
	"fmt"
	"text/tabwriter"
)

// Dump is a raw set of goroutines and their stacks.
type Dump []Goroutine

// Goroutine is a single parsed goroutine dump.
type Goroutine struct {
	State string // state that the goroutine is in.
	ID    int    // id of the goroutine.
	Stack Stack  // call frames that make up the stack
}

// Stack is a set of frames in a callstack.
type Stack []Frame

// Frame is a point in a call stack.
type Frame struct {
	Function Function
	Position Position
}

// Function is the function called at a frame.
type Function struct {
	Package string // package name of function if known
	Type    string // if set function is a method of this type
	Name    string // function name of the frame
}

// Position is the file position for a frame.
type Position struct {
	Filename string // source filename
	Line     int    // line number within file
}

// Summary is a set of stacks processed and collated into Calls.
type Summary struct {
	Total int    // the total count of goroutines in the summary
	Calls []Call // the collated stack traces
}

// Call is set of goroutines that all share the same callstack.
// They will be grouped by state.
type Call struct {
	Stack  Stack   // the shared callstack information
	Groups []Group // the sets of goroutines with the same state
}

// Group is a set of goroutines with the same stack that are in the same state.
type Group struct {
	State      string      // the shared state of the goroutines
	Goroutines []Goroutine // the set of goroutines in this group
}

// Delta represents the difference between two stack dumps.
type Delta struct {
	Before Dump // The goroutines that were only in the before set.
	Shared Dump // The goroutines that were in both sets.
	After  Dump // The goroutines that were only in the after set.
}

func (s Stack) equal(other Stack) bool {
	if len(s) != len(other) {
		return false
	}
	for i, frame := range s {
		if !frame.equal(other[i]) {
			return false
		}
	}
	return true
}

func (s Stack) less(other Stack) bool {
	for i, frame := range s {
		if i >= len(other) {
			return false
		}
		if frame.less(other[i]) {
			return true
		}
		if !frame.equal(other[i]) {
			return false
		}
	}
	return len(s) < len(other)
}

func (f Frame) equal(other Frame) bool {
	return f.Position.equal(other.Position)
}

func (f Frame) less(other Frame) bool {
	return f.Position.less(other.Position)
}

func (p Position) equal(other Position) bool {
	return p.Filename == other.Filename && p.Line == other.Line
}

func (p Position) less(other Position) bool {
	if p.Filename < other.Filename {
		return true
	}
	if p.Filename > other.Filename {
		return false
	}
	return p.Line < other.Line
}

func (s Summary) Format(w fmt.State, r rune) {
	tw := tabwriter.NewWriter(w, 0, 0, 1, ' ', 0)
	for i, c := range s.Calls {
		if i > 0 {
			fmt.Fprintf(tw, "\n\n")
			tw.Flush()
		}
		fmt.Fprint(tw, c)
	}
	tw.Flush()
	if s.Total > 0 && w.Flag('+') {
		fmt.Fprintf(w, "\n\n%d goroutines, %d unique", s.Total, len(s.Calls))
	}
}

func (c Call) Format(w fmt.State, r rune) {
	for i, g := range c.Groups {
		if i > 0 {
			fmt.Fprint(w, " ")
		}
		fmt.Fprint(w, g)
	}
	for _, f := range c.Stack {
		fmt.Fprintf(w, "\n%v", f)
	}
}

func (g Group) Format(w fmt.State, r rune) {
	fmt.Fprintf(w, "[%v]: ", g.State)
	for i, gr := range g.Goroutines {
		if i > 0 {
			fmt.Fprint(w, ", ")
		}
		fmt.Fprintf(w, "$%d", gr.ID)
	}
}

func (f Frame) Format(w fmt.State, c rune) {
	fmt.Fprintf(w, "%v:\t%v", f.Position, f.Function)
}

func (f Function) Format(w fmt.State, c rune) {
	if f.Type != "" {
		fmt.Fprintf(w, "(%v).", f.Type)
	}
	fmt.Fprintf(w, "%v", f.Name)
}

func (p Position) Format(w fmt.State, c rune) {
	fmt.Fprintf(w, "%v:%v", p.Filename, p.Line)
}

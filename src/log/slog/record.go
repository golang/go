// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package slog

import (
	"iter"
	"runtime"
	"slices"
	"time"
)

const nAttrsInline = 5

// A Record holds information about a log event.
// Copies of a Record share state.
// Do not modify a Record after handing out a copy to it.
// Call [NewRecord] to create a new Record.
// Use [Record.Clone] to create a copy with no shared state.
type Record struct {
	// The time at which the output method (Log, Info, etc.) was called.
	Time time.Time

	// The log message.
	Message string

	// The level of the event.
	Level Level

	// The program counter at the time the record was constructed, as determined
	// by runtime.Callers. If zero, no program counter is available.
	//
	// The only valid use for this value is as an argument to
	// [runtime.CallersFrames]. In particular, it must not be passed to
	// [runtime.FuncForPC].
	PC uintptr

	// Allocation optimization: an inline array sized to hold
	// the majority of log calls (based on examination of open-source
	// code). It holds the start of the list of Attrs.
	front [nAttrsInline]Attr

	// The number of Attrs in front.
	nFront int

	// The list of Attrs except for those in front.
	// Invariants:
	//   - len(back) > 0 iff nFront == len(front)
	//   - Unused array elements are zero. Used to detect mistakes.
	back []Attr
}

// NewRecord creates a [Record] from the given arguments.
// Use [Record.AddAttrs] to add attributes to the Record.
//
// NewRecord is intended for logging APIs that want to support a [Handler] as
// a backend.
func NewRecord(t time.Time, level Level, msg string, pc uintptr) Record {
	return Record{
		Time:    t,
		Message: msg,
		Level:   level,
		PC:      pc,
	}
}

// Clone returns a copy of the record with no shared state.
// The original record and the clone can both be modified
// without interfering with each other.
func (r Record) Clone() Record {
	r.back = slices.Clip(r.back) // prevent append from mutating shared array
	return r
}

// NumAttrs returns the number of attributes in the [Record].
func (r Record) NumAttrs() int {
	return r.nFront + len(r.back)
}

// Attrs calls f on each Attr in the [Record].
// Iteration stops if f returns false.
func (r Record) Attrs(f func(Attr) bool) {
	for i := 0; i < r.nFront; i++ {
		if !f(r.front[i]) {
			return
		}
	}
	for _, a := range r.back {
		if !f(a) {
			return
		}
	}
}

// AttrsIter returns an iterator over every attribute in the [Record].
func (r Record) AttrsIter() iter.Seq[Attr] {
	return func(yield func(Attr) bool) {
		r.Attrs(yield)
	}
}

// AddAttrs appends the given Attrs to the [Record]'s list of Attrs.
// It omits empty groups.
func (r *Record) AddAttrs(attrs ...Attr) {
	var i int
	for i = 0; i < len(attrs) && r.nFront < len(r.front); i++ {
		a := attrs[i]
		if a.Value.isEmptyGroup() {
			continue
		}
		r.front[r.nFront] = a
		r.nFront++
	}
	// Check if a copy was modified by slicing past the end
	// and seeing if the Attr there is non-zero.
	if cap(r.back) > len(r.back) {
		end := r.back[:len(r.back)+1][len(r.back)]
		if !end.isEmpty() {
			// Don't panic; copy and muddle through.
			r.back = slices.Clip(r.back)
			r.back = append(r.back, String("!BUG", "AddAttrs unsafely called on copy of Record made without using Record.Clone"))
		}
	}
	ne := countEmptyGroups(attrs[i:])
	r.back = slices.Grow(r.back, len(attrs[i:])-ne)
	for _, a := range attrs[i:] {
		if !a.Value.isEmptyGroup() {
			r.back = append(r.back, a)
		}
	}
}

// Add converts the args to Attrs as described in [Logger.Log],
// then appends the Attrs to the [Record]'s list of Attrs.
// It omits empty groups.
func (r *Record) Add(args ...any) {
	var a Attr
	for len(args) > 0 {
		a, args = argsToAttr(args)
		if a.Value.isEmptyGroup() {
			continue
		}
		if r.nFront < len(r.front) {
			r.front[r.nFront] = a
			r.nFront++
		} else {
			if r.back == nil {
				r.back = make([]Attr, 0, countAttrs(args)+1)
			}
			r.back = append(r.back, a)
		}
	}
}

// countAttrs returns the number of Attrs that would be created from args.
func countAttrs(args []any) int {
	n := 0
	for i := 0; i < len(args); i++ {
		n++
		if _, ok := args[i].(string); ok {
			i++
		}
	}
	return n
}

const badKey = "!BADKEY"

// argsToAttr turns a prefix of the nonempty args slice into an Attr
// and returns the unconsumed portion of the slice.
// If args[0] is an Attr, it returns it.
// If args[0] is a string, it treats the first two elements as
// a key-value pair.
// Otherwise, it treats args[0] as a value with a missing key.
func argsToAttr(args []any) (Attr, []any) {
	switch x := args[0].(type) {
	case string:
		if len(args) == 1 {
			return String(badKey, x), nil
		}
		return Any(x, args[1]), args[2:]

	case Attr:
		return x, args[1:]

	default:
		return Any(badKey, x), args[1:]
	}
}

// Source describes the location of a line of source code.
type Source struct {
	// Function is the package path-qualified function name containing the
	// source line. If non-empty, this string uniquely identifies a single
	// function in the program. This may be the empty string if not known.
	Function string `json:"function"`
	// File and Line are the file name and line number (1-based) of the source
	// line. These may be the empty string and zero, respectively, if not known.
	File string `json:"file"`
	Line int    `json:"line"`
}

// group returns the non-zero fields of s as a slice of attrs.
// It is similar to a LogValue method, but we don't want Source
// to implement LogValuer because it would be resolved before
// the ReplaceAttr function was called.
func (s *Source) group() Value {
	var as []Attr
	if s.Function != "" {
		as = append(as, String("function", s.Function))
	}
	if s.File != "" {
		as = append(as, String("file", s.File))
	}
	if s.Line != 0 {
		as = append(as, Int("line", s.Line))
	}
	return GroupValue(as...)
}

// isEmpty returns whether the Source struct is nil or only contains zero fields.
func (s *Source) isEmpty() bool { return s == nil || *s == Source{} }

// Source returns a new Source for the log event using r's PC.
// If the PC field is zero, meaning the Record was created without the necessary information
// or the location is unavailable, then nil is returned.
func (r Record) Source() *Source {
	if r.PC == 0 {
		return nil
	}

	fs := runtime.CallersFrames([]uintptr{r.PC})
	f, _ := fs.Next()
	return &Source{
		Function: f.Function,
		File:     f.File,
		Line:     f.Line,
	}
}

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package scanner

import (
	"fmt"
	"go/token"
	"io"
	"sort"
)

// An implementation of an ErrorHandler may be provided to the Scanner.
// If a syntax error is encountered and a handler was installed, Error
// is called with a position and an error message. The position points
// to the beginning of the offending token.
//
type ErrorHandler interface {
	Error(pos token.Position, msg string)
}

// ErrorVector implements the ErrorHandler interface. It maintains a list
// of errors which can be retrieved with GetErrorList and GetError. The
// zero value for an ErrorVector is an empty ErrorVector ready to use.
//
// A common usage pattern is to embed an ErrorVector alongside a
// scanner in a data structure that uses the scanner. By passing a
// reference to an ErrorVector to the scanner's Init call, default
// error handling is obtained.
//
type ErrorVector struct {
	errors []*Error
}

// Reset resets an ErrorVector to no errors.
func (h *ErrorVector) Reset() { h.errors = h.errors[:0] }

// ErrorCount returns the number of errors collected.
func (h *ErrorVector) ErrorCount() int { return len(h.errors) }

// Within ErrorVector, an error is represented by an Error node. The
// position Pos, if valid, points to the beginning of the offending
// token, and the error condition is described by Msg.
//
type Error struct {
	Pos token.Position
	Msg string
}

func (e *Error) Error() string {
	if e.Pos.Filename != "" || e.Pos.IsValid() {
		// don't print "<unknown position>"
		// TODO(gri) reconsider the semantics of Position.IsValid
		return e.Pos.String() + ": " + e.Msg
	}
	return e.Msg
}

// An ErrorList is a (possibly sorted) list of Errors.
type ErrorList []*Error

// ErrorList implements the sort Interface.
func (p ErrorList) Len() int      { return len(p) }
func (p ErrorList) Swap(i, j int) { p[i], p[j] = p[j], p[i] }

func (p ErrorList) Less(i, j int) bool {
	e := &p[i].Pos
	f := &p[j].Pos
	// Note that it is not sufficient to simply compare file offsets because
	// the offsets do not reflect modified line information (through //line
	// comments).
	if e.Filename < f.Filename {
		return true
	}
	if e.Filename == f.Filename {
		if e.Line < f.Line {
			return true
		}
		if e.Line == f.Line {
			return e.Column < f.Column
		}
	}
	return false
}

func (p ErrorList) Error() string {
	switch len(p) {
	case 0:
		return "unspecified error"
	case 1:
		return p[0].Error()
	}
	return fmt.Sprintf("%s (and %d more errors)", p[0], len(p)-1)
}

// These constants control the construction of the ErrorList
// returned by GetErrors.
//
const (
	Raw         = iota // leave error list unchanged
	Sorted             // sort error list by file, line, and column number
	NoMultiples        // sort error list and leave only the first error per line
)

// GetErrorList returns the list of errors collected by an ErrorVector.
// The construction of the ErrorList returned is controlled by the mode
// parameter. If there are no errors, the result is nil.
//
func (h *ErrorVector) GetErrorList(mode int) ErrorList {
	if len(h.errors) == 0 {
		return nil
	}

	list := make(ErrorList, len(h.errors))
	copy(list, h.errors)

	if mode >= Sorted {
		sort.Sort(list)
	}

	if mode >= NoMultiples {
		var last token.Position // initial last.Line is != any legal error line
		i := 0
		for _, e := range list {
			if e.Pos.Filename != last.Filename || e.Pos.Line != last.Line {
				last = e.Pos
				list[i] = e
				i++
			}
		}
		list = list[0:i]
	}

	return list
}

// GetError is like GetErrorList, but it returns an error instead
// so that a nil result can be assigned to an error variable and
// remains nil.
//
func (h *ErrorVector) GetError(mode int) error {
	if len(h.errors) == 0 {
		return nil
	}

	return h.GetErrorList(mode)
}

// ErrorVector implements the ErrorHandler interface.
func (h *ErrorVector) Error(pos token.Position, msg string) {
	h.errors = append(h.errors, &Error{pos, msg})
}

// PrintError is a utility function that prints a list of errors to w,
// one error per line, if the err parameter is an ErrorList. Otherwise
// it prints the err string.
//
func PrintError(w io.Writer, err error) {
	if list, ok := err.(ErrorList); ok {
		for _, e := range list {
			fmt.Fprintf(w, "%s\n", e)
		}
	} else {
		fmt.Fprintf(w, "%s\n", err)
	}
}

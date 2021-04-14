// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package xerrors

import (
	"fmt"
	"strings"
	"unicode"
	"unicode/utf8"

	"golang.org/x/xerrors/internal"
)

const percentBangString = "%!"

// Errorf formats according to a format specifier and returns the string as a
// value that satisfies error.
//
// The returned error includes the file and line number of the caller when
// formatted with additional detail enabled. If the last argument is an error
// the returned error's Format method will return it if the format string ends
// with ": %s", ": %v", or ": %w". If the last argument is an error and the
// format string ends with ": %w", the returned error implements an Unwrap
// method returning it.
//
// If the format specifier includes a %w verb with an error operand in a
// position other than at the end, the returned error will still implement an
// Unwrap method returning the operand, but the error's Format method will not
// return the wrapped error.
//
// It is invalid to include more than one %w verb or to supply it with an
// operand that does not implement the error interface. The %w verb is otherwise
// a synonym for %v.
func Errorf(format string, a ...interface{}) error {
	format = formatPlusW(format)
	// Support a ": %[wsv]" suffix, which works well with xerrors.Formatter.
	wrap := strings.HasSuffix(format, ": %w")
	idx, format2, ok := parsePercentW(format)
	percentWElsewhere := !wrap && idx >= 0
	if !percentWElsewhere && (wrap || strings.HasSuffix(format, ": %s") || strings.HasSuffix(format, ": %v")) {
		err := errorAt(a, len(a)-1)
		if err == nil {
			return &noWrapError{fmt.Sprintf(format, a...), nil, Caller(1)}
		}
		// TODO: this is not entirely correct. The error value could be
		// printed elsewhere in format if it mixes numbered with unnumbered
		// substitutions. With relatively small changes to doPrintf we can
		// have it optionally ignore extra arguments and pass the argument
		// list in its entirety.
		msg := fmt.Sprintf(format[:len(format)-len(": %s")], a[:len(a)-1]...)
		frame := Frame{}
		if internal.EnableTrace {
			frame = Caller(1)
		}
		if wrap {
			return &wrapError{msg, err, frame}
		}
		return &noWrapError{msg, err, frame}
	}
	// Support %w anywhere.
	// TODO: don't repeat the wrapped error's message when %w occurs in the middle.
	msg := fmt.Sprintf(format2, a...)
	if idx < 0 {
		return &noWrapError{msg, nil, Caller(1)}
	}
	err := errorAt(a, idx)
	if !ok || err == nil {
		// Too many %ws or argument of %w is not an error. Approximate the Go
		// 1.13 fmt.Errorf message.
		return &noWrapError{fmt.Sprintf("%sw(%s)", percentBangString, msg), nil, Caller(1)}
	}
	frame := Frame{}
	if internal.EnableTrace {
		frame = Caller(1)
	}
	return &wrapError{msg, err, frame}
}

func errorAt(args []interface{}, i int) error {
	if i < 0 || i >= len(args) {
		return nil
	}
	err, ok := args[i].(error)
	if !ok {
		return nil
	}
	return err
}

// formatPlusW is used to avoid the vet check that will barf at %w.
func formatPlusW(s string) string {
	return s
}

// Return the index of the only %w in format, or -1 if none.
// Also return a rewritten format string with %w replaced by %v, and
// false if there is more than one %w.
// TODO: handle "%[N]w".
func parsePercentW(format string) (idx int, newFormat string, ok bool) {
	// Loosely copied from golang.org/x/tools/go/analysis/passes/printf/printf.go.
	idx = -1
	ok = true
	n := 0
	sz := 0
	var isW bool
	for i := 0; i < len(format); i += sz {
		if format[i] != '%' {
			sz = 1
			continue
		}
		// "%%" is not a format directive.
		if i+1 < len(format) && format[i+1] == '%' {
			sz = 2
			continue
		}
		sz, isW = parsePrintfVerb(format[i:])
		if isW {
			if idx >= 0 {
				ok = false
			} else {
				idx = n
			}
			// "Replace" the last character, the 'w', with a 'v'.
			p := i + sz - 1
			format = format[:p] + "v" + format[p+1:]
		}
		n++
	}
	return idx, format, ok
}

// Parse the printf verb starting with a % at s[0].
// Return how many bytes it occupies and whether the verb is 'w'.
func parsePrintfVerb(s string) (int, bool) {
	// Assume only that the directive is a sequence of non-letters followed by a single letter.
	sz := 0
	var r rune
	for i := 1; i < len(s); i += sz {
		r, sz = utf8.DecodeRuneInString(s[i:])
		if unicode.IsLetter(r) {
			return i + sz, r == 'w'
		}
	}
	return len(s), false
}

type noWrapError struct {
	msg   string
	err   error
	frame Frame
}

func (e *noWrapError) Error() string {
	return fmt.Sprint(e)
}

func (e *noWrapError) Format(s fmt.State, v rune) { FormatError(e, s, v) }

func (e *noWrapError) FormatError(p Printer) (next error) {
	p.Print(e.msg)
	e.frame.Format(p)
	return e.err
}

type wrapError struct {
	msg   string
	err   error
	frame Frame
}

func (e *wrapError) Error() string {
	return fmt.Sprint(e)
}

func (e *wrapError) Format(s fmt.State, v rune) { FormatError(e, s, v) }

func (e *wrapError) FormatError(p Printer) (next error) {
	p.Print(e.msg)
	e.frame.Format(p)
	return e.err
}

func (e *wrapError) Unwrap() error {
	return e.err
}

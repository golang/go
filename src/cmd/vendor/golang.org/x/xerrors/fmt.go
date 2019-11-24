// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package xerrors

import (
	"fmt"
	"strings"

	"golang.org/x/xerrors/internal"
)

// Errorf formats according to a format specifier and returns the string as a
// value that satisfies error.
//
// The returned error includes the file and line number of the caller when
// formatted with additional detail enabled. If the last argument is an error
// the returned error's Format method will return it if the format string ends
// with ": %s", ": %v", or ": %w". If the last argument is an error and the
// format string ends with ": %w", the returned error implements Wrapper
// with an Unwrap method returning it.
func Errorf(format string, a ...interface{}) error {
	err, wrap := lastError(format, a)
	format = formatPlusW(format)
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

// formatPlusW is used to avoid the vet check that will barf at %w.
func formatPlusW(s string) string {
	return s
}

func lastError(format string, a []interface{}) (err error, wrap bool) {
	wrap = strings.HasSuffix(format, ": %w")
	if !wrap &&
		!strings.HasSuffix(format, ": %s") &&
		!strings.HasSuffix(format, ": %v") {
		return nil, false
	}

	if len(a) == 0 {
		return nil, false
	}

	err, ok := a[len(a)-1].(error)
	if !ok {
		return nil, false
	}

	return err, wrap
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

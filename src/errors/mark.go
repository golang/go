// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package errors

import (
	"fmt"
	"io"
)

var _ error = markError{} // verify that Error implements error

// Mark returns an error with the supplied errors as marks.
// If err is nil, return nil.
// marks take effects only when Is and '%v' in fmt.
// Is returns true if err or any marks match the target.
func Mark(err error, marks ...error) error {
	if err == nil {
		return nil
	}
	if len(marks) == 0 {
		return err
	}
	me := markError{
		err:   err,
		marks: marks,
	}
	return me
}

type markError struct {
	err   error   // visual error
	marks []error // hidden errors as marks, take effects only when Is and '%v' in fmt.
}

func (e markError) Error() string {
	if e.err == nil {
		return ""
	}
	return e.err.Error()
}

func (e markError) Format(s fmt.State, verb rune) {
	if e.err == nil {
		return
	}
	switch verb {
	case 'v':
		if s.Flag('+') {
			me := e.clean()
			if len(me.marks) == 0 {
				_, _ = fmt.Fprintf(s, "%+v", me.err)
				return
			}
			_, _ = io.WriteString(s, "Marked errors occurred:\n")

			_, _ = fmt.Fprintf(s, "|\t%+v", me.err)
			for _, mark := range me.marks {
				_, _ = fmt.Fprintf(s, "\nM\t%+v", mark)
			}
			return
		}
		fallthrough
	case 's', 'q':
		_, _ = io.WriteString(s, e.Error())
	}
}

// clean removes all none nil elem in all the marks
func (e markError) clean() markError {
	var marks []error
	for _, err := range e.marks {
		if err != nil {
			marks = append(marks, err)
		}
	}
	return markError{
		err:   e.err,
		marks: marks,
	}
}

// Is reports whether any error in markError or it's mark errors matches target.
func (e markError) Is(target error) bool {
	if Is(e.err, target) {
		return true
	}
	for _, err := range e.marks {
		if Is(err, target) {
			return true
		}
	}
	return false
}

// Unwrap returns the error in e, if there is exactly one. If there is more than one
// error, Unwrap returns nil, since there is no way to determine which should be
// returned.
func (e markError) Unwrap() error {
	return e.err
}

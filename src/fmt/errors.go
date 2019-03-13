// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fmt

import (
	"errors"
	"internal/errinternal"
	"strings"
)

// Errorf formats according to a format specifier and returns the string as a
// value that satisfies error.
//
// The returned error includes the file and line number of the caller when
// formatted with additional detail enabled. If the last argument is an error
// the returned error's Format method will return it if the format string ends
// with ": %s", ": %v", or ": %w". If the last argument is an error and the
// format string ends with ": %w", the returned error implements errors.Wrapper
// with an Unwrap method returning it.
func Errorf(format string, a ...interface{}) error {
	err, wrap := lastError(format, a)
	if err == nil {
		return errinternal.NewError(Sprintf(format, a...), nil)
	}

	// TODO: this is not entirely correct. The error value could be
	// printed elsewhere in format if it mixes numbered with unnumbered
	// substitutions. With relatively small changes to doPrintf we can
	// have it optionally ignore extra arguments and pass the argument
	// list in its entirety.
	msg := Sprintf(format[:len(format)-len(": %s")], a[:len(a)-1]...)
	if wrap {
		return &wrapError{msg, err, errors.Caller(1)}
	}
	return errinternal.NewError(msg, err)
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
	frame errors.Frame
}

func (e *noWrapError) Error() string {
	return Sprint(e)
}

func (e *noWrapError) FormatError(p errors.Printer) (next error) {
	p.Print(e.msg)
	e.frame.Format(p)
	return e.err
}

type wrapError struct {
	msg   string
	err   error
	frame errors.Frame
}

func (e *wrapError) Error() string {
	return Sprint(e)
}

func (e *wrapError) FormatError(p errors.Printer) (next error) {
	p.Print(e.msg)
	e.frame.Format(p)
	return e.err
}

func (e *wrapError) Unwrap() error {
	return e.err
}

func fmtError(p *pp, verb rune, err error) (handled bool) {
	var (
		sep = " " // separator before next error
		w   = p   // print buffer where error text is written
	)
	switch {
	// Note that this switch must match the preference order
	// for ordinary string printing (%#v before %+v, and so on).

	case p.fmt.sharpV:
		if stringer, ok := p.arg.(GoStringer); ok {
			// Print the result of GoString unadorned.
			p.fmt.fmtS(stringer.GoString())
			return true
		}
		return false

	case p.fmt.plusV:
		sep = "\n  - "
		w.fmt.fmtFlags = fmtFlags{plusV: p.fmt.plusV} // only keep detail flag

		// The width or precision of a detailed view could be the number of
		// errors to print from a list.

	default:
		// Use an intermediate buffer in the rare cases that precision,
		// truncation, or one of the alternative verbs (q, x, and X) are
		// specified.
		switch verb {
		case 's', 'v':
			if (!w.fmt.widPresent || w.fmt.wid == 0) && !w.fmt.precPresent {
				break
			}
			fallthrough
		case 'q', 'x', 'X':
			w = newPrinter()
			defer w.free()
		default:
			return false
		}
	}

loop:
	for {
		w.fmt.inDetail = false
		switch v := err.(type) {
		case errors.Formatter:
			err = v.FormatError((*errPP)(w))
		case Formatter:
			if w.fmt.plusV {
				v.Format((*errPPState)(w), 'v') // indent new lines
			} else {
				v.Format(w, 'v') // do not indent new lines
			}
			break loop
		default:
			w.fmtString(v.Error(), 's')
			break loop
		}
		if err == nil {
			break
		}
		if w.fmt.needColon || !p.fmt.plusV {
			w.buf.WriteByte(':')
			w.fmt.needColon = false
		}
		w.buf.WriteString(sep)
		w.fmt.inDetail = false
		w.fmt.needNewline = false
	}

	if w != p {
		p.fmtString(string(w.buf), verb)
	}
	return true
}

var detailSep = []byte("\n    ")

// errPPState wraps a pp to implement State with indentation. It is used
// for errors implementing fmt.Formatter.
type errPPState pp

func (p *errPPState) Width() (wid int, ok bool)      { return (*pp)(p).Width() }
func (p *errPPState) Precision() (prec int, ok bool) { return (*pp)(p).Precision() }
func (p *errPPState) Flag(c int) bool                { return (*pp)(p).Flag(c) }

func (p *errPPState) Write(b []byte) (n int, err error) {
	if p.fmt.plusV {
		if len(b) == 0 {
			return 0, nil
		}
		if p.fmt.inDetail && p.fmt.needColon {
			p.fmt.needNewline = true
			if b[0] == '\n' {
				b = b[1:]
			}
		}
		k := 0
		for i, c := range b {
			if p.fmt.needNewline {
				if p.fmt.inDetail && p.fmt.needColon {
					p.buf.WriteByte(':')
					p.fmt.needColon = false
				}
				p.buf.Write(detailSep)
				p.fmt.needNewline = false
			}
			if c == '\n' {
				p.buf.Write(b[k:i])
				k = i + 1
				p.fmt.needNewline = true
			}
		}
		p.buf.Write(b[k:])
		if !p.fmt.inDetail {
			p.fmt.needColon = true
		}
	} else if !p.fmt.inDetail {
		p.buf.Write(b)
	}
	return len(b), nil

}

// errPP wraps a pp to implement an errors.Printer.
type errPP pp

func (p *errPP) Print(args ...interface{}) {
	if !p.fmt.inDetail || p.fmt.plusV {
		Fprint((*errPPState)(p), args...)
	}
}

func (p *errPP) Printf(format string, args ...interface{}) {
	if !p.fmt.inDetail || p.fmt.plusV {
		Fprintf((*errPPState)(p), format, args...)
	}
}

func (p *errPP) Detail() bool {
	p.fmt.inDetail = true
	return p.fmt.plusV
}

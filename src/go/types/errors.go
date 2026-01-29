// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements error reporting.

package types

import (
	"fmt"
	"go/ast"
	"go/token"
	. "internal/types/errors"
	"runtime"
	"strings"
)

func assert(p bool) {
	if !p {
		msg := "assertion failed"
		// Include information about the assertion location. Due to panic recovery,
		// this location is otherwise buried in the middle of the panicking stack.
		if _, file, line, ok := runtime.Caller(1); ok {
			msg = fmt.Sprintf("%s:%d: %s", file, line, msg)
		}
		panic(msg)
	}
}

// An errorDesc describes part of a type-checking error.
type errorDesc struct {
	posn positioner
	msg  string
}

// An error_ represents a type-checking error.
// A new error_ is created with Checker.newError.
// To report an error_, call error_.report.
type error_ struct {
	check *Checker
	desc  []errorDesc
	code  Code
	soft  bool // TODO(gri) eventually determine this from an error code
}

// newError returns a new error_ with the given error code.
func (check *Checker) newError(code Code) *error_ {
	if code == 0 {
		panic("error code must not be 0")
	}
	return &error_{check: check, code: code}
}

// addf adds formatted error information to err.
// It may be called multiple times to provide additional information.
// The position of the first call to addf determines the position of the reported Error.
// Subsequent calls to addf provide additional information in the form of additional lines
// in the error message (types2) or continuation errors identified by a tab-indented error
// message (go/types).
func (err *error_) addf(at positioner, format string, args ...any) {
	err.desc = append(err.desc, errorDesc{at, err.check.sprintf(format, args...)})
}

// addAltDecl is a specialized form of addf reporting another declaration of obj.
func (err *error_) addAltDecl(obj Object) {
	if pos := obj.Pos(); pos.IsValid() {
		// We use "other" rather than "previous" here because
		// the first declaration seen may not be textually
		// earlier in the source.
		err.addf(obj, "other declaration of %s", obj.Name())
	}
}

func (err *error_) empty() bool {
	return err.desc == nil
}

func (err *error_) posn() positioner {
	if err.empty() {
		return noposn
	}
	return err.desc[0].posn
}

// msg returns the formatted error message without the primary error position pos().
func (err *error_) msg() string {
	if err.empty() {
		return "no error"
	}

	var buf strings.Builder
	for i := range err.desc {
		p := &err.desc[i]
		if i > 0 {
			fmt.Fprint(&buf, "\n\t")
			if p.posn.Pos().IsValid() {
				fmt.Fprintf(&buf, "%s: ", err.check.fset.Position(p.posn.Pos()))
			}
		}
		buf.WriteString(p.msg)
	}
	return buf.String()
}

// report reports the error err, setting check.firstError if necessary.
func (err *error_) report() {
	if err.empty() {
		panic("no error")
	}

	// Cheap trick: Don't report errors with messages containing
	// "invalid operand" or "invalid type" as those tend to be
	// follow-on errors which don't add useful information. Only
	// exclude them if these strings are not at the beginning,
	// and only if we have at least one error already reported.
	check := err.check
	if check.firstErr != nil {
		// It is sufficient to look at the first sub-error only.
		msg := err.desc[0].msg
		if strings.Index(msg, "invalid operand") > 0 || strings.Index(msg, "invalid type") > 0 {
			return
		}
	}

	if check.conf._Trace {
		check.trace(err.posn().Pos(), "ERROR: %s (code = %d)", err.desc[0].msg, err.code)
	}

	// In go/types, if there is a sub-error with a valid position,
	// call the typechecker error handler for each sub-error.
	// Otherwise, call it once, with a single combined message.
	multiError := false
	if !isTypes2 {
		for i := 1; i < len(err.desc); i++ {
			if err.desc[i].posn.Pos().IsValid() {
				multiError = true
				break
			}
		}
	}

	if multiError {
		for i := range err.desc {
			p := &err.desc[i]
			check.handleError(i, p.posn, err.code, p.msg, err.soft)
		}
	} else {
		check.handleError(0, err.posn(), err.code, err.msg(), err.soft)
	}

	// make sure the error is not reported twice
	err.desc = nil
}

// handleError should only be called by error_.report.
func (check *Checker) handleError(index int, posn positioner, code Code, msg string, soft bool) {
	assert(code != 0)

	if index == 0 {
		// If we are encountering an error while evaluating an inherited
		// constant initialization expression, pos is the position of
		// the original expression, and not of the currently declared
		// constant identifier. Use the provided errpos instead.
		// TODO(gri) We may also want to augment the error message and
		// refer to the position (pos) in the original expression.
		if check.errpos != nil && check.errpos.Pos().IsValid() {
			assert(check.iota != nil)
			posn = check.errpos
		}

		// Report invalid syntax trees explicitly.
		if code == InvalidSyntaxTree {
			msg = "invalid syntax tree: " + msg
		}

		// If we have a URL for error codes, add a link to the first line.
		if check.conf._ErrorURL != "" {
			url := fmt.Sprintf(check.conf._ErrorURL, code)
			if i := strings.Index(msg, "\n"); i >= 0 {
				msg = msg[:i] + url + msg[i:]
			} else {
				msg += url
			}
		}
	} else {
		// Indent sub-error.
		// Position information is passed explicitly to Error, below.
		msg = "\t" + msg
	}

	span := spanOf(posn)
	e := Error{
		Fset:       check.fset,
		Pos:        span.pos,
		Msg:        stripAnnotations(msg),
		Soft:       soft,
		go116code:  code,
		go116start: span.start,
		go116end:   span.end,
	}

	if check.errpos != nil {
		// If we have an internal error and the errpos override is set, use it to
		// augment our error positioning.
		// TODO(rFindley) we may also want to augment the error message and refer
		// to the position (pos) in the original expression.
		span := spanOf(check.errpos)
		e.Pos = span.pos
		e.go116start = span.start
		e.go116end = span.end
	}

	if check.firstErr == nil {
		check.firstErr = e
	}

	f := check.conf.Error
	if f == nil {
		panic(bailout{}) // record first error and exit
	}
	f(e)
}

const (
	invalidArg = "invalid argument: "
	invalidOp  = "invalid operation: "
)

// The positioner interface is used to extract the position of type-checker errors.
type positioner interface {
	Pos() token.Pos
}

func (check *Checker) error(at positioner, code Code, msg string) {
	err := check.newError(code)
	err.addf(at, "%s", msg)
	err.report()
}

func (check *Checker) errorf(at positioner, code Code, format string, args ...any) {
	err := check.newError(code)
	err.addf(at, format, args...)
	err.report()
}

func (check *Checker) softErrorf(at positioner, code Code, format string, args ...any) {
	err := check.newError(code)
	err.addf(at, format, args...)
	err.soft = true
	err.report()
}

func (check *Checker) versionErrorf(at positioner, v goVersion, format string, args ...any) {
	msg := check.sprintf(format, args...)
	err := check.newError(UnsupportedFeature)
	err.addf(at, "%s requires %s or later", msg, v)
	err.report()
}

// atPos wraps a token.Pos to implement the positioner interface.
type atPos token.Pos

func (s atPos) Pos() token.Pos {
	return token.Pos(s)
}

// posSpan holds a position range along with a highlighted position within that
// range. This is used for positioning errors, with pos by convention being the
// first position in the source where the error is known to exist, and start
// and end defining the full span of syntax being considered when the error was
// detected. Invariant: start <= pos < end || start == pos == end.
type posSpan struct {
	start, pos, end token.Pos
}

func (e posSpan) Pos() token.Pos {
	return e.pos
}

// inNode creates a posSpan for the given node.
// Invariant: node.Pos() <= pos < node.End() (node.End() is the position of the
// first byte after node within the source).
func inNode(node ast.Node, pos token.Pos) posSpan {
	start, end := node.Pos(), node.End()
	if debug {
		assert(start <= pos && pos < end)
	}
	return posSpan{start, pos, end}
}

// spanOf extracts an error span from the given positioner. By default this is
// the trivial span starting and ending at pos, but this span is expanded when
// the argument naturally corresponds to a span of source code.
func spanOf(at positioner) posSpan {
	switch x := at.(type) {
	case nil:
		panic("nil positioner")
	case posSpan:
		return x
	case ast.Node:
		pos := x.Pos()
		return posSpan{pos, pos, x.End()}
	case *operand:
		if x.expr != nil {
			pos := x.Pos()
			return posSpan{pos, pos, x.expr.End()}
		}
		return posSpan{nopos, nopos, nopos}
	default:
		pos := at.Pos()
		return posSpan{pos, pos, pos}
	}
}

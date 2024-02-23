// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements various error reporters.

package types

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/token"
	. "internal/types/errors"
	"runtime"
	"strconv"
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

// An error_ represents a type-checking error.
// To report an error_, call Checker.report.
type error_ struct {
	desc []errorDesc
	code Code
	soft bool // TODO(gri) eventually determine this from an error code
}

// An errorDesc describes part of a type-checking error.
type errorDesc struct {
	posn   positioner
	format string
	args   []interface{}
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

func (err *error_) msg(fset *token.FileSet, qf Qualifier) string {
	if err.empty() {
		return "no error"
	}
	var buf strings.Builder
	for i := range err.desc {
		p := &err.desc[i]
		if i > 0 {
			fmt.Fprint(&buf, "\n\t")
			if p.posn.Pos().IsValid() {
				fmt.Fprintf(&buf, "%s: ", fset.Position(p.posn.Pos()))
			}
		}
		buf.WriteString(sprintf(fset, qf, false, p.format, p.args...))
	}
	return buf.String()
}

// String is for testing.
func (err *error_) String() string {
	if err.empty() {
		return "no error"
	}
	return fmt.Sprintf("%d: %s", err.posn().Pos(), err.msg(nil, nil))
}

// errorf adds formatted error information to err.
// It may be called multiple times to provide additional information.
func (err *error_) errorf(at positioner, format string, args ...interface{}) {
	err.desc = append(err.desc, errorDesc{at, format, args})
}

func (check *Checker) qualifier(pkg *Package) string {
	// Qualify the package unless it's the package being type-checked.
	if pkg != check.pkg {
		if check.pkgPathMap == nil {
			check.pkgPathMap = make(map[string]map[string]bool)
			check.seenPkgMap = make(map[*Package]bool)
			check.markImports(check.pkg)
		}
		// If the same package name was used by multiple packages, display the full path.
		if len(check.pkgPathMap[pkg.name]) > 1 {
			return strconv.Quote(pkg.path)
		}
		return pkg.name
	}
	return ""
}

// markImports recursively walks pkg and its imports, to record unique import
// paths in pkgPathMap.
func (check *Checker) markImports(pkg *Package) {
	if check.seenPkgMap[pkg] {
		return
	}
	check.seenPkgMap[pkg] = true

	forName, ok := check.pkgPathMap[pkg.name]
	if !ok {
		forName = make(map[string]bool)
		check.pkgPathMap[pkg.name] = forName
	}
	forName[pkg.path] = true

	for _, imp := range pkg.imports {
		check.markImports(imp)
	}
}

// check may be nil.
func (check *Checker) sprintf(format string, args ...any) string {
	var fset *token.FileSet
	var qf Qualifier
	if check != nil {
		fset = check.fset
		qf = check.qualifier
	}
	return sprintf(fset, qf, false, format, args...)
}

func sprintf(fset *token.FileSet, qf Qualifier, tpSubscripts bool, format string, args ...any) string {
	for i, arg := range args {
		switch a := arg.(type) {
		case nil:
			arg = "<nil>"
		case operand:
			panic("got operand instead of *operand")
		case *operand:
			arg = operandString(a, qf)
		case token.Pos:
			if fset != nil {
				arg = fset.Position(a).String()
			}
		case ast.Expr:
			arg = ExprString(a)
		case []ast.Expr:
			var buf bytes.Buffer
			buf.WriteByte('[')
			writeExprList(&buf, a)
			buf.WriteByte(']')
			arg = buf.String()
		case Object:
			arg = ObjectString(a, qf)
		case Type:
			var buf bytes.Buffer
			w := newTypeWriter(&buf, qf)
			w.tpSubscripts = tpSubscripts
			w.typ(a)
			arg = buf.String()
		case []Type:
			var buf bytes.Buffer
			w := newTypeWriter(&buf, qf)
			w.tpSubscripts = tpSubscripts
			buf.WriteByte('[')
			for i, x := range a {
				if i > 0 {
					buf.WriteString(", ")
				}
				w.typ(x)
			}
			buf.WriteByte(']')
			arg = buf.String()
		case []*TypeParam:
			var buf bytes.Buffer
			w := newTypeWriter(&buf, qf)
			w.tpSubscripts = tpSubscripts
			buf.WriteByte('[')
			for i, x := range a {
				if i > 0 {
					buf.WriteString(", ")
				}
				w.typ(x)
			}
			buf.WriteByte(']')
			arg = buf.String()
		}
		args[i] = arg
	}
	return fmt.Sprintf(format, args...)
}

func (check *Checker) trace(pos token.Pos, format string, args ...any) {
	fmt.Printf("%s:\t%s%s\n",
		check.fset.Position(pos),
		strings.Repeat(".  ", check.indent),
		sprintf(check.fset, check.qualifier, true, format, args...),
	)
}

// dump is only needed for debugging
func (check *Checker) dump(format string, args ...any) {
	fmt.Println(sprintf(check.fset, check.qualifier, true, format, args...))
}

// Report records the error pointed to by errp, setting check.firstError if
// necessary.
func (check *Checker) report(errp *error_) {
	if errp.empty() {
		panic("empty error details")
	}

	msg := errp.msg(check.fset, check.qualifier)
	switch errp.code {
	case InvalidSyntaxTree:
		msg = "invalid AST: " + msg
	case 0:
		panic("no error code provided")
	}

	// If we have a URL for error codes, add a link to the first line.
	if errp.code != 0 && check.conf._ErrorURL != "" {
		u := fmt.Sprintf(check.conf._ErrorURL, errp.code)
		if i := strings.Index(msg, "\n"); i >= 0 {
			msg = msg[:i] + u + msg[i:]
		} else {
			msg += u
		}
	}

	span := spanOf(errp.desc[0].posn)
	e := Error{
		Fset:       check.fset,
		Pos:        span.pos,
		Msg:        msg,
		Soft:       errp.soft,
		go116code:  errp.code,
		go116start: span.start,
		go116end:   span.end,
	}

	// Cheap trick: Don't report errors with messages containing
	// "invalid operand" or "invalid type" as those tend to be
	// follow-on errors which don't add useful information. Only
	// exclude them if these strings are not at the beginning,
	// and only if we have at least one error already reported.
	isInvalidErr := strings.Index(e.Msg, "invalid operand") > 0 || strings.Index(e.Msg, "invalid type") > 0
	if check.firstErr != nil && isInvalidErr {
		return
	}

	e.Msg = stripAnnotations(e.Msg)
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
	err := e

	if check.firstErr == nil {
		check.firstErr = err
	}

	if check.conf._Trace {
		pos := e.Pos
		msg := e.Msg
		check.trace(pos, "ERROR: %s", msg)
	}

	f := check.conf.Error
	if f == nil {
		panic(bailout{}) // report only first error
	}
	f(err)
}

const (
	invalidArg = "invalid argument: "
	invalidOp  = "invalid operation: "
)

// newErrorf creates a new error_ for later reporting with check.report.
func newErrorf(at positioner, code Code, format string, args ...any) *error_ {
	return &error_{
		desc: []errorDesc{{at, format, args}},
		code: code,
	}
}

func (check *Checker) error(at positioner, code Code, msg string) {
	check.report(newErrorf(at, code, "%s", msg))
}

func (check *Checker) errorf(at positioner, code Code, format string, args ...any) {
	check.report(newErrorf(at, code, format, args...))
}

func (check *Checker) softErrorf(at positioner, code Code, format string, args ...any) {
	err := newErrorf(at, code, format, args...)
	err.soft = true
	check.report(err)
}

func (check *Checker) versionErrorf(at positioner, v goVersion, format string, args ...interface{}) {
	msg := check.sprintf(format, args...)
	var err *error_
	err = newErrorf(at, UnsupportedFeature, "%s requires %s or later", msg, v)
	check.report(err)
}

// The positioner interface is used to extract the position of type-checker
// errors.
type positioner interface {
	Pos() token.Pos
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

// atPos wraps a token.Pos to implement the positioner interface.
type atPos token.Pos

func (s atPos) Pos() token.Pos {
	return token.Pos(s)
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

// stripAnnotations removes internal (type) annotations from s.
func stripAnnotations(s string) string {
	var buf strings.Builder
	for _, r := range s {
		// strip #'s and subscript digits
		if r < '₀' || '₀'+10 <= r { // '₀' == U+2080
			buf.WriteRune(r)
		}
	}
	if buf.Len() < len(s) {
		return buf.String()
	}
	return s
}

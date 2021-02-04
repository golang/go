// UNREVIEWED
// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements various error reporters.

package types2

import (
	"bytes"
	"cmd/compile/internal/syntax"
	"fmt"
	"strconv"
	"strings"
)

func unimplemented() {
	panic("unimplemented")
}

func assert(p bool) {
	if !p {
		panic("assertion failed")
	}
}

func unreachable() {
	panic("unreachable")
}

func (check *Checker) qualifier(pkg *Package) string {
	// Qualify the package unless it's the package being type-checked.
	if pkg != check.pkg {
		// If the same package name was used by multiple packages, display the full path.
		if check.pkgCnt[pkg.name] > 1 {
			return strconv.Quote(pkg.path)
		}
		return pkg.name
	}
	return ""
}

func (check *Checker) sprintf(format string, args ...interface{}) string {
	for i, arg := range args {
		switch a := arg.(type) {
		case nil:
			arg = "<nil>"
		case operand:
			panic("internal error: should always pass *operand")
		case *operand:
			arg = operandString(a, check.qualifier)
		case syntax.Pos:
			arg = a.String()
		case syntax.Expr:
			arg = syntax.String(a)
		case Object:
			arg = ObjectString(a, check.qualifier)
		case Type:
			arg = TypeString(a, check.qualifier)
		}
		args[i] = arg
	}
	return fmt.Sprintf(format, args...)
}

func (check *Checker) trace(pos syntax.Pos, format string, args ...interface{}) {
	fmt.Printf("%s:\t%s%s\n",
		pos,
		strings.Repeat(".  ", check.indent),
		check.sprintf(format, args...),
	)
}

// dump is only needed for debugging
func (check *Checker) dump(format string, args ...interface{}) {
	fmt.Println(check.sprintf(format, args...))
}

func (check *Checker) err(pos syntax.Pos, msg string, soft bool) {
	// Cheap trick: Don't report errors with messages containing
	// "invalid operand" or "invalid type" as those tend to be
	// follow-on errors which don't add useful information. Only
	// exclude them if these strings are not at the beginning,
	// and only if we have at least one error already reported.
	if check.firstErr != nil && (strings.Index(msg, "invalid operand") > 0 || strings.Index(msg, "invalid type") > 0) {
		return
	}

	// If we are encountering an error while evaluating an inherited
	// constant initialization expression, pos is the position of in
	// the original expression, and not of the currently declared
	// constant identifier. Use the provided errpos instead.
	// TODO(gri) We may also want to augment the error message and
	// refer to the position (pos) in the original expression.
	if check.errpos.IsKnown() {
		assert(check.iota != nil)
		pos = check.errpos
	}

	err := Error{pos, stripAnnotations(msg), msg, soft}
	if check.firstErr == nil {
		check.firstErr = err
	}

	if check.conf.Trace {
		check.trace(pos, "ERROR: %s", msg)
	}

	f := check.conf.Error
	if f == nil {
		panic(bailout{}) // report only first error
	}
	f(err)
}

type poser interface {
	Pos() syntax.Pos
}

func (check *Checker) error(at poser, msg string) {
	check.err(posFor(at), msg, false)
}

func (check *Checker) errorf(at poser, format string, args ...interface{}) {
	check.err(posFor(at), check.sprintf(format, args...), false)
}

func (check *Checker) softErrorf(at poser, format string, args ...interface{}) {
	check.err(posFor(at), check.sprintf(format, args...), true)
}

func (check *Checker) invalidASTf(at poser, format string, args ...interface{}) {
	check.errorf(at, "invalid AST: "+format, args...)
}

func (check *Checker) invalidArgf(at poser, format string, args ...interface{}) {
	check.errorf(at, "invalid argument: "+format, args...)
}

func (check *Checker) invalidOpf(at poser, format string, args ...interface{}) {
	check.errorf(at, "invalid operation: "+format, args...)
}

// posFor reports the left (= start) position of at.
func posFor(at poser) syntax.Pos {
	switch x := at.(type) {
	case *operand:
		if x.expr != nil {
			return startPos(x.expr)
		}
	case syntax.Node:
		return startPos(x)
	}
	return at.Pos()
}

// stripAnnotations removes internal (type) annotations from s.
func stripAnnotations(s string) string {
	// Would like to use strings.Builder but it's not available in Go 1.4.
	var b bytes.Buffer
	for _, r := range s {
		// strip #'s and subscript digits
		if r != instanceMarker && !('₀' <= r && r < '₀'+10) { // '₀' == U+2080
			b.WriteRune(r)
		}
	}
	if b.Len() < len(s) {
		return b.String()
	}
	return s
}

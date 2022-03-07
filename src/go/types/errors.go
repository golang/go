// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements various error reporters.

package types

import (
	"bytes"
	"errors"
	"fmt"
	"go/ast"
	"go/token"
	"strconv"
	"strings"
)

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

func sprintf(fset *token.FileSet, qf Qualifier, debug bool, format string, args ...any) string {
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
			arg = typeString(a, qf, debug)
		case []Type:
			var buf bytes.Buffer
			buf.WriteByte('[')
			for i, x := range a {
				if i > 0 {
					buf.WriteString(", ")
				}
				buf.WriteString(typeString(x, qf, debug))
			}
			buf.WriteByte(']')
			arg = buf.String()
		case []*TypeParam:
			var buf bytes.Buffer
			buf.WriteByte('[')
			for i, x := range a {
				if i > 0 {
					buf.WriteString(", ")
				}
				buf.WriteString(typeString(x, qf, debug)) // use typeString so we get subscripts when debugging
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

func (check *Checker) err(err error) {
	if err == nil {
		return
	}
	var e Error
	isInternal := errors.As(err, &e)
	// Cheap trick: Don't report errors with messages containing
	// "invalid operand" or "invalid type" as those tend to be
	// follow-on errors which don't add useful information. Only
	// exclude them if these strings are not at the beginning,
	// and only if we have at least one error already reported.
	isInvalidErr := isInternal && (strings.Index(e.Msg, "invalid operand") > 0 || strings.Index(e.Msg, "invalid type") > 0)
	if check.firstErr != nil && isInvalidErr {
		return
	}

	if isInternal {
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
		err = e
	}

	if check.firstErr == nil {
		check.firstErr = err
	}

	if trace {
		pos := e.Pos
		msg := e.Msg
		if !isInternal {
			msg = err.Error()
			pos = token.NoPos
		}
		check.trace(pos, "ERROR: %s", msg)
	}

	f := check.conf.Error
	if f == nil {
		panic(bailout{}) // report only first error
	}
	f(err)
}

func (check *Checker) newError(at positioner, code errorCode, soft bool, msg string) error {
	span := spanOf(at)
	return Error{
		Fset:       check.fset,
		Pos:        span.pos,
		Msg:        msg,
		Soft:       soft,
		go116code:  code,
		go116start: span.start,
		go116end:   span.end,
	}
}

// newErrorf creates a new Error, but does not handle it.
func (check *Checker) newErrorf(at positioner, code errorCode, soft bool, format string, args ...any) error {
	msg := check.sprintf(format, args...)
	return check.newError(at, code, soft, msg)
}

func (check *Checker) error(at positioner, code errorCode, msg string) {
	check.err(check.newError(at, code, false, msg))
}

func (check *Checker) errorf(at positioner, code errorCode, format string, args ...any) {
	check.error(at, code, check.sprintf(format, args...))
}

func (check *Checker) softErrorf(at positioner, code errorCode, format string, args ...any) {
	check.err(check.newErrorf(at, code, true, format, args...))
}

func (check *Checker) invalidAST(at positioner, format string, args ...any) {
	check.errorf(at, 0, "invalid AST: "+format, args...)
}

func (check *Checker) invalidArg(at positioner, code errorCode, format string, args ...any) {
	check.errorf(at, code, "invalid argument: "+format, args...)
}

func (check *Checker) invalidOp(at positioner, code errorCode, format string, args ...any) {
	check.errorf(at, code, "invalid operation: "+format, args...)
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
		return posSpan{token.NoPos, token.NoPos, token.NoPos}
	default:
		pos := at.Pos()
		return posSpan{pos, pos, pos}
	}
}

// stripAnnotations removes internal (type) annotations from s.
func stripAnnotations(s string) string {
	var b strings.Builder
	for _, r := range s {
		// strip #'s and subscript digits
		if r < '₀' || '₀'+10 <= r { // '₀' == U+2080
			b.WriteRune(r)
		}
	}
	if b.Len() < len(s) {
		return b.String()
	}
	return s
}

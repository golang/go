// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements various error reporters.

package types2

import (
	"cmd/compile/internal/syntax"
	"fmt"
	"runtime"
	"strconv"
	"strings"
)

func unimplemented() {
	panic("unimplemented")
}

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

func unreachable() {
	panic("unreachable")
}

// An error_ represents a type-checking error.
// To report an error_, call Checker.report.
type error_ struct {
	desc []errorDesc
	soft bool // TODO(gri) eventually determine this from an error code
}

// An errorDesc describes part of a type-checking error.
type errorDesc struct {
	pos    syntax.Pos
	format string
	args   []interface{}
}

func (err *error_) empty() bool {
	return err.desc == nil
}

func (err *error_) pos() syntax.Pos {
	if err.empty() {
		return nopos
	}
	return err.desc[0].pos
}

func (err *error_) msg(qf Qualifier) string {
	if err.empty() {
		return "no error"
	}
	var buf strings.Builder
	for i := range err.desc {
		p := &err.desc[i]
		if i > 0 {
			fmt.Fprint(&buf, "\n\t")
			if p.pos.IsKnown() {
				fmt.Fprintf(&buf, "%s: ", p.pos)
			}
		}
		buf.WriteString(sprintf(qf, false, p.format, p.args...))
	}
	return buf.String()
}

// String is for testing.
func (err *error_) String() string {
	if err.empty() {
		return "no error"
	}
	return fmt.Sprintf("%s: %s", err.pos(), err.msg(nil))
}

// errorf adds formatted error information to err.
// It may be called multiple times to provide additional information.
func (err *error_) errorf(at poser, format string, args ...interface{}) {
	err.desc = append(err.desc, errorDesc{posFor(at), format, args})
}

func sprintf(qf Qualifier, debug bool, format string, args ...interface{}) string {
	for i, arg := range args {
		switch a := arg.(type) {
		case nil:
			arg = "<nil>"
		case operand:
			panic("got operand instead of *operand")
		case *operand:
			arg = operandString(a, qf)
		case syntax.Pos:
			arg = a.String()
		case syntax.Expr:
			arg = syntax.String(a)
		case []syntax.Expr:
			var buf strings.Builder
			buf.WriteByte('[')
			for i, x := range a {
				if i > 0 {
					buf.WriteString(", ")
				}
				buf.WriteString(syntax.String(x))
			}
			buf.WriteByte(']')
			arg = buf.String()
		case Object:
			arg = ObjectString(a, qf)
		case Type:
			arg = typeString(a, qf, debug)
		case []Type:
			var buf strings.Builder
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
			var buf strings.Builder
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
func (check *Checker) sprintf(format string, args ...interface{}) string {
	var qf Qualifier
	if check != nil {
		qf = check.qualifier
	}
	return sprintf(qf, false, format, args...)
}

func (check *Checker) report(err *error_) {
	if err.empty() {
		panic("no error to report")
	}
	check.err(err.pos(), err.msg(check.qualifier), err.soft)
}

func (check *Checker) trace(pos syntax.Pos, format string, args ...interface{}) {
	fmt.Printf("%s:\t%s%s\n",
		pos,
		strings.Repeat(".  ", check.indent),
		sprintf(check.qualifier, true, format, args...),
	)
}

// dump is only needed for debugging
func (check *Checker) dump(format string, args ...interface{}) {
	fmt.Println(sprintf(check.qualifier, true, format, args...))
}

func (check *Checker) err(at poser, msg string, soft bool) {
	// Cheap trick: Don't report errors with messages containing
	// "invalid operand" or "invalid type" as those tend to be
	// follow-on errors which don't add useful information. Only
	// exclude them if these strings are not at the beginning,
	// and only if we have at least one error already reported.
	if check.firstErr != nil && (strings.Index(msg, "invalid operand") > 0 || strings.Index(msg, "invalid type") > 0) {
		return
	}

	pos := posFor(at)

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

const (
	invalidAST = "invalid AST: "
	invalidArg = "invalid argument: "
	invalidOp  = "invalid operation: "
)

type poser interface {
	Pos() syntax.Pos
}

func (check *Checker) error(at poser, msg string) {
	check.err(at, msg, false)
}

func (check *Checker) errorf(at poser, format string, args ...interface{}) {
	check.err(at, check.sprintf(format, args...), false)
}

func (check *Checker) softErrorf(at poser, format string, args ...interface{}) {
	check.err(at, check.sprintf(format, args...), true)
}

func (check *Checker) versionErrorf(at poser, goVersion string, format string, args ...interface{}) {
	msg := check.sprintf(format, args...)
	if check.conf.CompilerErrorMessages {
		msg = fmt.Sprintf("%s requires %s or later (-lang was set to %s; check go.mod)", msg, goVersion, check.conf.GoVersion)
	} else {
		msg = fmt.Sprintf("%s requires %s or later", msg, goVersion)
	}
	check.err(at, msg, true)
}

// posFor reports the left (= start) position of at.
func posFor(at poser) syntax.Pos {
	switch x := at.(type) {
	case *operand:
		if x.expr != nil {
			return syntax.StartPos(x.expr)
		}
	case syntax.Node:
		return syntax.StartPos(x)
	}
	return at.Pos()
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

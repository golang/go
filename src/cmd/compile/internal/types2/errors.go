// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements various error reporters.

package types2

import (
	"bytes"
	"cmd/compile/internal/syntax"
	"fmt"
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
			if p.pos.IsKnown() {
				fmt.Fprintf(&buf, "%s: ", p.pos)
			}
		}
		buf.WriteString(err.check.sprintf(p.format, p.args...))
	}
	return buf.String()
}

// addf adds formatted error information to err.
// It may be called multiple times to provide additional information.
// The position of the first call to addf determines the position of the reported Error.
// Subsequent calls to addf provide additional information in the form of additional lines
// in the error message (types2) or continuation errors identified by a tab-indented error
// message (go/types).
func (err *error_) addf(at poser, format string, args ...interface{}) {
	err.desc = append(err.desc, errorDesc{atPos(at), format, args})
}

func sprintf(qf Qualifier, tpSubscripts bool, format string, args ...interface{}) string {
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
			arg = ExprString(a)
		case []syntax.Expr:
			var buf strings.Builder
			buf.WriteByte('[')
			for i, x := range a {
				if i > 0 {
					buf.WriteString(", ")
				}
				buf.WriteString(ExprString(x))
			}
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

// report reports the error err, setting check.firstError if necessary.
func (err *error_) report() {
	if err.empty() {
		panic("no error to report")
	}

	msg := err.msg()
	code := err.code
	assert(code != 0)
	if code == InvalidSyntaxTree {
		msg = "invalid syntax tree: " + msg
	}

	// If we are encountering an error while evaluating an inherited
	// constant initialization expression, pos is the position of in
	// the original expression, and not of the currently declared
	// constant identifier. Use the provided errpos instead.
	// TODO(gri) We may also want to augment the error message and
	// refer to the position (pos) in the original expression.
	check := err.check
	pos := err.pos()
	if check.errpos.IsKnown() {
		assert(check.iota != nil)
		pos = check.errpos
	}

	if check.conf.Trace {
		check.trace(pos, "ERROR: %s", msg)
	}

	// Cheap trick: Don't report errors with messages containing
	// "invalid operand" or "invalid type" as those tend to be
	// follow-on errors which don't add useful information. Only
	// exclude them if these strings are not at the beginning,
	// and only if we have at least one error already reported.
	isInvalidErr := strings.Index(msg, "invalid operand") > 0 || strings.Index(msg, "invalid type") > 0
	if check.firstErr != nil && isInvalidErr {
		return
	}

	// If we have a URL for error codes, add a link to the first line.
	if check.conf.ErrorURL != "" {
		u := fmt.Sprintf(check.conf.ErrorURL, code)
		if i := strings.Index(msg, "\n"); i >= 0 {
			msg = msg[:i] + u + msg[i:]
		} else {
			msg += u
		}
	}

	e := Error{
		Pos:  pos,
		Msg:  stripAnnotations(msg),
		Full: msg,
		Soft: err.soft,
		Code: code,
	}

	if check.firstErr == nil {
		check.firstErr = e
	}

	f := check.conf.Error
	if f == nil {
		panic(bailout{}) // report only first error
	}

	// TODO(gri) If e contains \t-indented sub-errors,
	//           for go/types f must be called for each
	//           of those sub-errors.
	f(e)
}

const (
	invalidArg = "invalid argument: "
	invalidOp  = "invalid operation: "
)

type poser interface {
	Pos() syntax.Pos
}

func (check *Checker) error(at poser, code Code, msg string) {
	err := check.newError(code)
	err.addf(at, "%s", msg)
	err.report()
}

func (check *Checker) errorf(at poser, code Code, format string, args ...interface{}) {
	err := check.newError(code)
	err.addf(at, format, args...)
	err.report()
}

func (check *Checker) softErrorf(at poser, code Code, format string, args ...interface{}) {
	err := check.newError(code)
	err.addf(at, format, args...)
	err.soft = true
	err.report()
}

func (check *Checker) versionErrorf(at poser, v goVersion, format string, args ...interface{}) {
	msg := check.sprintf(format, args...)
	err := check.newError(UnsupportedFeature)
	err.addf(at, "%s requires %s or later", msg, v)
	err.report()
}

// atPos reports the left (= start) position of at.
func atPos(at poser) syntax.Pos {
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

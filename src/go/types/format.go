// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements (error and trace) message formatting support.

package types

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/token"
	"strconv"
	"strings"
)

// quote encloses s in `' quotes, as in `foo', except for _,
// which is left alone.
//
// Use to prevent confusion when user supplied names alter the
// meaning of an error message.
//
// For instance, report
//
//	duplicate method `wanted'
//
// rather than
//
//	duplicate method wanted
//
// Exceptions:
//
//   - don't quote _:
//     `_' is ugly and not necessary
//   - don't quote after a ":" as there's no need for it:
//     undefined name: foo
//   - don't quote if the name is used correctly in a statement:
//     goto L jumps over variable declaration
//
// quote encloses s in `' quotes, as in `foo',
// except for _ which is left alone.
func quote(s string) string {
	if s == "_" {
		// `_' is ugly and not necessary
		return s
	}
	return "`" + s + "'"
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

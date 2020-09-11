// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements various error reporters.

package types2

import (
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
			arg = ExprString(a)
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

func (check *Checker) error(at interface{}, msg string) {
	check.err(posFor(at), msg, false)
}

func (check *Checker) errorf(at interface{}, format string, args ...interface{}) {
	check.err(posFor(at), check.sprintf(format, args...), false)
}

func (check *Checker) softErrorf(at interface{}, format string, args ...interface{}) {
	check.err(posFor(at), check.sprintf(format, args...), true)
}

func (check *Checker) invalidASTf(at interface{}, format string, args ...interface{}) {
	check.errorf(at, "invalid AST: "+format, args...)
}

func (check *Checker) invalidArgf(at interface{}, format string, args ...interface{}) {
	check.errorf(at, "invalid argument: "+format, args...)
}

func (check *Checker) invalidOpf(at interface{}, format string, args ...interface{}) {
	check.errorf(at, "invalid operation: "+format, args...)
}

// posFor reports the left (= start) position of at.
func posFor(at interface{}) syntax.Pos {
	switch x := at.(type) {
	case nil:
		panic("internal error: nil")
	case syntax.Pos:
		return x
	case operand:
		panic("internal error: should always pass *operand")
	case *operand:
		if x.expr != nil {
			return leftPos(x.expr)
		}
		return nopos
	case syntax.Node:
		return leftPos(x)
	case Object:
		return x.Pos()
	default:
		panic(fmt.Sprintf("internal error: unknown type %T", x))
	}
}

// leftPos returns left (= start) position of n.
func leftPos(n syntax.Node) (pos syntax.Pos) {
	// Cases for nodes which don't need a correction are commented out.
	switch n := n.(type) {
	case nil:
		panic("internal error: nil")

	// packages
	// case *syntax.File:

	// declarations
	// case *syntax.ImportDecl:
	// case *syntax.ConstDecl:
	// case *syntax.TypeDecl:
	// case *syntax.VarDecl:
	// case *syntax.FuncDecl:

	// expressions
	// case *syntax.BadExpr:
	// case *syntax.Name:
	// case *syntax.BasicLit:
	case *syntax.CompositeLit:
		if n.Type != nil {
			return leftPos(n.Type)
		}
	// case *syntax.KeyValueExpr:
	// case *syntax.FuncLit:
	// case *syntax.ParenExpr:
	case *syntax.SelectorExpr:
		return leftPos(n.X)
	case *syntax.IndexExpr:
		return leftPos(n.X)
	// case *syntax.SliceExpr:
	case *syntax.AssertExpr:
		return leftPos(n.X)
	case *syntax.TypeSwitchGuard:
		if n.Lhs != nil {
			return leftPos(n.Lhs)
		}
		return leftPos(n.X)
	case *syntax.Operation:
		if n.Y != nil {
			return leftPos(n.X)
		}
	case *syntax.CallExpr:
		return leftPos(n.Fun)
	case *syntax.ListExpr:
		if len(n.ElemList) > 0 {
			return leftPos(n.ElemList[0])
		}
	// types
	// case *syntax.ArrayType:
	// case *syntax.SliceType:
	// case *syntax.DotsType:
	// case *syntax.StructType:
	// case *syntax.Field:
	// case *syntax.InterfaceType:
	// case *syntax.FuncType:
	// case *syntax.MapType:
	// case *syntax.ChanType:

	// statements
	// case *syntax.EmptyStmt:
	// case *syntax.LabeledStmt:
	// case *syntax.BlockStmt:
	// case *syntax.ExprStmt:
	case *syntax.SendStmt:
		return leftPos(n.Chan)
	// case *syntax.DeclStmt:
	case *syntax.AssignStmt:
		return leftPos(n.Lhs)
	// case *syntax.BranchStmt:
	// case *syntax.CallStmt:
	// case *syntax.ReturnStmt:
	// case *syntax.IfStmt:
	// case *syntax.ForStmt:
	// case *syntax.SwitchStmt:
	// case *syntax.SelectStmt:

	// helper nodes
	case *syntax.RangeClause:
		if n.Lhs != nil {
			return leftPos(n.Lhs)
		}
		return leftPos(n.X)
		// case *syntax.CaseClause:
		// case *syntax.CommClause:
	}

	return n.Pos()
}

// stripAnnotations removes internal (type) annotations from s.
func stripAnnotations(s string) string {
	var b strings.Builder
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

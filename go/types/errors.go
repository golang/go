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
	"strings"
)

// TODO(gri) eventually assert should disappear.
func assert(p bool) {
	if !p {
		panic("assertion failed")
	}
}

func unreachable() {
	panic("unreachable")
}

func (check *checker) formatMsg(format string, args []interface{}) string {
	for i, arg := range args {
		switch a := arg.(type) {
		case nil:
			args[i] = "<nil>"
		case operand:
			panic("internal error: should always pass *operand")
		case token.Pos:
			args[i] = check.fset.Position(a)
		case ast.Expr:
			args[i] = exprString(a)
		}
	}
	return fmt.Sprintf(format, args...)
}

func (check *checker) trace(pos token.Pos, format string, args ...interface{}) {
	fmt.Printf("%s:\t%s%s\n",
		check.fset.Position(pos),
		strings.Repeat(".  ", check.indent),
		check.formatMsg(format, args),
	)
}

// dump is only needed for debugging
func (check *checker) dump(format string, args ...interface{}) {
	fmt.Println(check.formatMsg(format, args))
}

func (check *checker) err(err error) {
	if check.firstErr == nil {
		check.firstErr = err
	}
	f := check.conf.Error
	if f == nil {
		panic(bailout{}) // report only first error
	}
	f(err)
}

func (check *checker) errorf(pos token.Pos, format string, args ...interface{}) {
	check.err(fmt.Errorf("%s: %s", check.fset.Position(pos), check.formatMsg(format, args)))
}

func (check *checker) invalidAST(pos token.Pos, format string, args ...interface{}) {
	check.errorf(pos, "invalid AST: "+format, args...)
}

func (check *checker) invalidArg(pos token.Pos, format string, args ...interface{}) {
	check.errorf(pos, "invalid argument: "+format, args...)
}

func (check *checker) invalidOp(pos token.Pos, format string, args ...interface{}) {
	check.errorf(pos, "invalid operation: "+format, args...)
}

// exprString returns a (simplified) string representation for an expression.
func exprString(expr ast.Expr) string {
	var buf bytes.Buffer
	writeExpr(&buf, expr)
	return buf.String()
}

// TODO(gri) Need to merge with typeString since some expressions are types (try: ([]int)(a))
func writeExpr(buf *bytes.Buffer, expr ast.Expr) {
	switch x := expr.(type) {
	case *ast.Ident:
		buf.WriteString(x.Name)

	case *ast.BasicLit:
		buf.WriteString(x.Value)

	case *ast.FuncLit:
		buf.WriteString("(func literal)")

	case *ast.CompositeLit:
		buf.WriteString("(composite literal)")

	case *ast.ParenExpr:
		buf.WriteByte('(')
		writeExpr(buf, x.X)
		buf.WriteByte(')')

	case *ast.SelectorExpr:
		writeExpr(buf, x.X)
		buf.WriteByte('.')
		buf.WriteString(x.Sel.Name)

	case *ast.IndexExpr:
		writeExpr(buf, x.X)
		buf.WriteByte('[')
		writeExpr(buf, x.Index)
		buf.WriteByte(']')

	case *ast.SliceExpr:
		writeExpr(buf, x.X)
		buf.WriteByte('[')
		if x.Low != nil {
			writeExpr(buf, x.Low)
		}
		buf.WriteByte(':')
		if x.High != nil {
			writeExpr(buf, x.High)
		}
		buf.WriteByte(']')

	case *ast.TypeAssertExpr:
		writeExpr(buf, x.X)
		buf.WriteString(".(")
		// TODO(gri) expand writeExpr so that types are not handled by default case
		writeExpr(buf, x.Type)
		buf.WriteByte(')')

	case *ast.CallExpr:
		writeExpr(buf, x.Fun)
		buf.WriteByte('(')
		for i, arg := range x.Args {
			if i > 0 {
				buf.WriteString(", ")
			}
			writeExpr(buf, arg)
		}
		buf.WriteByte(')')

	case *ast.StarExpr:
		buf.WriteByte('*')
		writeExpr(buf, x.X)

	case *ast.UnaryExpr:
		buf.WriteString(x.Op.String())
		writeExpr(buf, x.X)

	case *ast.BinaryExpr:
		// The AST preserves source-level parentheses so there is
		// no need to introduce parentheses here for correctness.
		writeExpr(buf, x.X)
		buf.WriteByte(' ')
		buf.WriteString(x.Op.String())
		buf.WriteByte(' ')
		writeExpr(buf, x.Y)

	default:
		// TODO(gri) Consider just calling x.String(). May cause
		//           infinite recursion if we missed a local type.
		fmt.Fprintf(buf, "<expr %T>", x)
	}
}

// typeString returns a string representation for typ.
func typeString(typ Type) string {
	var buf bytes.Buffer
	writeType(&buf, typ)
	return buf.String()
}

func writeTuple(buf *bytes.Buffer, tup *Tuple, isVariadic bool) {
	buf.WriteByte('(')
	if tup != nil {
		for i, v := range tup.vars {
			if i > 0 {
				buf.WriteString(", ")
			}
			if v.name != "" {
				buf.WriteString(v.name)
				buf.WriteByte(' ')
			}
			typ := v.typ
			if isVariadic && i == len(tup.vars)-1 {
				buf.WriteString("...")
				typ = typ.(*Slice).elt
			}
			writeType(buf, typ)
		}
	}
	buf.WriteByte(')')
}

func writeSignature(buf *bytes.Buffer, sig *Signature) {
	writeTuple(buf, sig.params, sig.isVariadic)

	n := sig.results.Len()
	if n == 0 {
		// no result
		return
	}

	buf.WriteByte(' ')
	if n == 1 && sig.results.vars[0].name == "" {
		// single unnamed result
		writeType(buf, sig.results.vars[0].typ)
		return
	}

	// multiple or named result(s)
	writeTuple(buf, sig.results, false)
}

func writeType(buf *bytes.Buffer, typ Type) {
	switch t := typ.(type) {
	case nil:
		buf.WriteString("<nil>")

	case *Basic:
		buf.WriteString(t.name)

	case *Array:
		fmt.Fprintf(buf, "[%d]", t.len)
		writeType(buf, t.elt)

	case *Slice:
		buf.WriteString("[]")
		writeType(buf, t.elt)

	case *Struct:
		buf.WriteString("struct{")
		for i, f := range t.fields {
			if i > 0 {
				buf.WriteString("; ")
			}
			if !f.anonymous {
				buf.WriteString(f.name)
				buf.WriteByte(' ')
			}
			writeType(buf, f.typ)
			if tag := t.Tag(i); tag != "" {
				fmt.Fprintf(buf, " %q", tag)
			}
		}
		buf.WriteByte('}')

	case *Pointer:
		buf.WriteByte('*')
		writeType(buf, t.base)

	case *Tuple:
		writeTuple(buf, t, false)

	case *Signature:
		buf.WriteString("func")
		writeSignature(buf, t)

	case *Builtin:
		fmt.Fprintf(buf, "<type of %s>", t.name)

	case *Interface:
		buf.WriteString("interface{")
		for i, m := range t.methods {
			if i > 0 {
				buf.WriteString("; ")
			}
			buf.WriteString(m.name)
			writeSignature(buf, m.typ.(*Signature))
		}
		buf.WriteByte('}')

	case *Map:
		buf.WriteString("map[")
		writeType(buf, t.key)
		buf.WriteByte(']')
		writeType(buf, t.elt)

	case *Chan:
		var s string
		switch t.dir {
		case ast.SEND:
			s = "chan<- "
		case ast.RECV:
			s = "<-chan "
		default:
			s = "chan "
		}
		buf.WriteString(s)
		writeType(buf, t.elt)

	case *Named:
		s := "<Named w/o object>"
		if obj := t.obj; obj != nil {
			if obj.pkg != nil {
				// TODO(gri) Ideally we only want the qualification
				// if we are referring to a type that was imported;
				// but not when we are at the "top". We don't have
				// this information easily available here.
				buf.WriteString(obj.pkg.name)
				buf.WriteByte('.')
			}
			s = t.obj.name
		}
		buf.WriteString(s)

	default:
		// For externally defined implementations of Type.
		buf.WriteString(t.String())
	}
}

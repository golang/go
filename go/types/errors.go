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

func assert(p bool) {
	if !p {
		panic("assertion failed")
	}
}

func unreachable() {
	panic("unreachable")
}

func (check *checker) sprintf(format string, args ...interface{}) string {
	for i, arg := range args {
		switch a := arg.(type) {
		case nil:
			args[i] = "<nil>"
		case operand:
			panic("internal error: should always pass *operand")
		case token.Pos:
			args[i] = check.fset.Position(a).String()
		case ast.Expr:
			args[i] = ExprString(a)
		}
	}
	return fmt.Sprintf(format, args...)
}

func (check *checker) trace(pos token.Pos, format string, args ...interface{}) {
	fmt.Printf("%s:\t%s%s\n",
		check.fset.Position(pos),
		strings.Repeat(".  ", check.indent),
		check.sprintf(format, args...),
	)
}

// dump is only needed for debugging
func (check *checker) dump(format string, args ...interface{}) {
	fmt.Println(check.sprintf(format, args...))
}

func (check *checker) err(pos token.Pos, msg string) {
	err := Error{check.fset, pos, msg}
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
	check.err(pos, check.sprintf(format, args...))
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
func ExprString(expr ast.Expr) string {
	var buf bytes.Buffer
	WriteExpr(&buf, expr)
	return buf.String()
}

// TODO(gri) Need to merge with TypeString since some expressions are types (try: ([]int)(a))
func WriteExpr(buf *bytes.Buffer, expr ast.Expr) {
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
		WriteExpr(buf, x.X)
		buf.WriteByte(')')

	case *ast.SelectorExpr:
		WriteExpr(buf, x.X)
		buf.WriteByte('.')
		buf.WriteString(x.Sel.Name)

	case *ast.IndexExpr:
		WriteExpr(buf, x.X)
		buf.WriteByte('[')
		WriteExpr(buf, x.Index)
		buf.WriteByte(']')

	case *ast.SliceExpr:
		WriteExpr(buf, x.X)
		buf.WriteByte('[')
		if x.Low != nil {
			WriteExpr(buf, x.Low)
		}
		buf.WriteByte(':')
		if x.High != nil {
			WriteExpr(buf, x.High)
		}
		buf.WriteByte(']')

	case *ast.TypeAssertExpr:
		WriteExpr(buf, x.X)
		buf.WriteString(".(")
		// TODO(gri) expand WriteExpr so that types are not handled by default case
		WriteExpr(buf, x.Type)
		buf.WriteByte(')')

	case *ast.CallExpr:
		WriteExpr(buf, x.Fun)
		buf.WriteByte('(')
		for i, arg := range x.Args {
			if i > 0 {
				buf.WriteString(", ")
			}
			WriteExpr(buf, arg)
		}
		buf.WriteByte(')')

	case *ast.StarExpr:
		buf.WriteByte('*')
		WriteExpr(buf, x.X)

	case *ast.UnaryExpr:
		buf.WriteString(x.Op.String())
		WriteExpr(buf, x.X)

	case *ast.BinaryExpr:
		// The AST preserves source-level parentheses so there is
		// no need to introduce parentheses here for correctness.
		WriteExpr(buf, x.X)
		buf.WriteByte(' ')
		buf.WriteString(x.Op.String())
		buf.WriteByte(' ')
		WriteExpr(buf, x.Y)

	default:
		// TODO(gri) Consider just calling x.String(). May cause
		//           infinite recursion if we missed a local type.
		fmt.Fprintf(buf, "<expr %T>", x)
	}
}

// TypeString returns the string form of typ.
// Named types are printed package-qualified only
// if they do not belong to this package.
//
func TypeString(this *Package, typ Type) string {
	var buf bytes.Buffer
	writeType(&buf, this, typ)
	return buf.String()
}

func writeTuple(buf *bytes.Buffer, this *Package, tup *Tuple, isVariadic bool) {
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
				typ = typ.(*Slice).elem
			}
			writeType(buf, this, typ)
		}
	}
	buf.WriteByte(')')
}

func writeSignature(buf *bytes.Buffer, this *Package, sig *Signature) {
	writeTuple(buf, this, sig.params, sig.isVariadic)

	n := sig.results.Len()
	if n == 0 {
		// no result
		return
	}

	buf.WriteByte(' ')
	if n == 1 && sig.results.vars[0].name == "" {
		// single unnamed result
		writeType(buf, this, sig.results.vars[0].typ)
		return
	}

	// multiple or named result(s)
	writeTuple(buf, this, sig.results, false)
}

func writeType(buf *bytes.Buffer, this *Package, typ Type) {
	switch t := typ.(type) {
	case nil:
		buf.WriteString("<nil>")

	case *Basic:
		if t.kind == UnsafePointer {
			buf.WriteString("unsafe.")
		}
		buf.WriteString(t.name)

	case *Array:
		fmt.Fprintf(buf, "[%d]", t.len)
		writeType(buf, this, t.elem)

	case *Slice:
		buf.WriteString("[]")
		writeType(buf, this, t.elem)

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
			writeType(buf, this, f.typ)
			if tag := t.Tag(i); tag != "" {
				fmt.Fprintf(buf, " %q", tag)
			}
		}
		buf.WriteByte('}')

	case *Pointer:
		buf.WriteByte('*')
		writeType(buf, this, t.base)

	case *Tuple:
		writeTuple(buf, this, t, false)

	case *Signature:
		buf.WriteString("func")
		writeSignature(buf, this, t)

	case *Interface:
		// We write the source-level methods and embedded types rather
		// than the actual method set since resolved method signatures
		// may have non-printable cycles if parameters have anonymous
		// interface types that (directly or indirectly) embed the
		// current interface. For instance, consider the result type
		// of m:
		//
		//     type T interface{
		//         m() interface{ T }
		//     }
		//
		buf.WriteString("interface{")
		for i, m := range t.methods {
			if i > 0 {
				buf.WriteString("; ")
			}
			buf.WriteString(m.name)
			writeSignature(buf, this, m.typ.(*Signature))
		}
		for i, typ := range t.types {
			if i > 0 || len(t.methods) > 0 {
				buf.WriteString("; ")
			}
			writeType(buf, this, typ)
		}
		buf.WriteByte('}')

	case *Map:
		buf.WriteString("map[")
		writeType(buf, this, t.key)
		buf.WriteByte(']')
		writeType(buf, this, t.elem)

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
		writeType(buf, this, t.elem)

	case *Named:
		s := "<Named w/o object>"
		if obj := t.obj; obj != nil {
			if obj.pkg != nil {
				if obj.pkg != this {
					buf.WriteString(obj.pkg.path)
					buf.WriteByte('.')
				}
				// TODO(gri) Ideally we only want the qualification
				// if we are referring to a type that was imported;
				// but not when we are at the "top". We don't have
				// this information easily available here.

				// Some applications may want another variant that accepts a
				// file Scope and prints packages using that file's local
				// import names.  (Printing just pkg.name may be ambiguous
				// or incorrect in other scopes.)

				// TODO(gri): function-local named types should be displayed
				// differently from named types at package level to avoid
				// ambiguity.
			}
			s = t.obj.name
		}
		buf.WriteString(s)

	default:
		// For externally defined implementations of Type.
		buf.WriteString(t.String())
	}
}

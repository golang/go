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
)

// TODO(gri) eventually assert and unimplemented should disappear.
func assert(p bool) {
	if !p {
		panic("assertion failed")
	}
}

func unreachable() {
	panic("unreachable")
}

func (check *checker) printTrace(format string, args []interface{}) {
	const dots = ".  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  "
	n := len(check.pos) - 1
	i := 3 * n
	for i > len(dots) {
		fmt.Print(dots)
		i -= len(dots)
	}
	// i <= len(dots)
	fmt.Printf("%s:\t", check.fset.Position(check.pos[n]))
	fmt.Print(dots[0:i])
	fmt.Println(check.formatMsg(format, args))
}

func (check *checker) trace(pos token.Pos, format string, args ...interface{}) {
	check.pos = append(check.pos, pos)
	check.printTrace(format, args)
}

func (check *checker) untrace(format string, args ...interface{}) {
	if len(format) > 0 {
		check.printTrace(format, args)
	}
	check.pos = check.pos[:len(check.pos)-1]
}

func (check *checker) formatMsg(format string, args []interface{}) string {
	for i, arg := range args {
		switch a := arg.(type) {
		case token.Pos:
			args[i] = check.fset.Position(a).String()
		case ast.Expr:
			args[i] = exprString(a)
		case Type:
			args[i] = typeString(a)
		case operand:
			panic("internal error: should always pass *operand")
		}
	}
	return fmt.Sprintf(format, args...)
}

// dump is only needed for debugging
func (check *checker) dump(format string, args ...interface{}) {
	fmt.Println(check.formatMsg(format, args))
}

func (check *checker) err(err error) {
	if check.firsterr == nil {
		check.firsterr = err
	}
	f := check.ctxt.Error
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
		buf.WriteString(".(...)")

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
		fmt.Fprintf(buf, "<expr %T>", x)
	}
}

// typeString returns a string representation for typ.
func typeString(typ Type) string {
	var buf bytes.Buffer
	writeType(&buf, typ)
	return buf.String()
}

func writeParams(buf *bytes.Buffer, params []*Var, isVariadic bool) {
	buf.WriteByte('(')
	for i, par := range params {
		if i > 0 {
			buf.WriteString(", ")
		}
		if par.Name != "" {
			buf.WriteString(par.Name)
			buf.WriteByte(' ')
		}
		if isVariadic && i == len(params)-1 {
			buf.WriteString("...")
		}
		writeType(buf, par.Type)
	}
	buf.WriteByte(')')
}

func writeSignature(buf *bytes.Buffer, sig *Signature) {
	writeParams(buf, sig.Params, sig.IsVariadic)
	if len(sig.Results) == 0 {
		// no result
		return
	}

	buf.WriteByte(' ')
	if len(sig.Results) == 1 && sig.Results[0].Name == "" {
		// single unnamed result
		writeType(buf, sig.Results[0].Type.(Type))
		return
	}

	// multiple or named result(s)
	writeParams(buf, sig.Results, false)
}

func writeType(buf *bytes.Buffer, typ Type) {
	switch t := typ.(type) {
	case nil:
		buf.WriteString("<nil>")

	case *Basic:
		buf.WriteString(t.Name)

	case *Array:
		fmt.Fprintf(buf, "[%d]", t.Len)
		writeType(buf, t.Elt)

	case *Slice:
		buf.WriteString("[]")
		writeType(buf, t.Elt)

	case *Struct:
		buf.WriteString("struct{")
		for i, f := range t.Fields {
			if i > 0 {
				buf.WriteString("; ")
			}
			if !f.IsAnonymous {
				buf.WriteString(f.Name)
				buf.WriteByte(' ')
			}
			writeType(buf, f.Type)
			if f.Tag != "" {
				fmt.Fprintf(buf, " %q", f.Tag)
			}
		}
		buf.WriteByte('}')

	case *Pointer:
		buf.WriteByte('*')
		writeType(buf, t.Base)

	case *Result:
		writeParams(buf, t.Values, false)

	case *Signature:
		buf.WriteString("func")
		writeSignature(buf, t)

	case *builtin:
		fmt.Fprintf(buf, "<type of %s>", t.name)

	case *Interface:
		buf.WriteString("interface{")
		for i, m := range t.Methods {
			if i > 0 {
				buf.WriteString("; ")
			}
			buf.WriteString(m.Name)
			writeSignature(buf, m.Type)
		}
		buf.WriteByte('}')

	case *Map:
		buf.WriteString("map[")
		writeType(buf, t.Key)
		buf.WriteByte(']')
		writeType(buf, t.Elt)

	case *Chan:
		var s string
		switch t.Dir {
		case ast.SEND:
			s = "chan<- "
		case ast.RECV:
			s = "<-chan "
		default:
			s = "chan "
		}
		buf.WriteString(s)
		writeType(buf, t.Elt)

	case *NamedType:
		s := "<NamedType w/o object>"
		if obj := t.Obj; obj != nil {
			if obj.Pkg != nil && obj.Pkg.Path != "" {
				buf.WriteString(obj.Pkg.Path)
				buf.WriteString(".")
			}
			s = t.Obj.GetName()
		}
		buf.WriteString(s)

	default:
		fmt.Fprintf(buf, "<type %T>", t)
	}
}

func (t *Array) String() string     { return typeString(t) }
func (t *Basic) String() string     { return typeString(t) }
func (t *Chan) String() string      { return typeString(t) }
func (t *Interface) String() string { return typeString(t) }
func (t *Map) String() string       { return typeString(t) }
func (t *NamedType) String() string { return typeString(t) }
func (t *Pointer) String() string   { return typeString(t) }
func (t *Result) String() string    { return typeString(t) }
func (t *Signature) String() string { return typeString(t) }
func (t *Slice) String() string     { return typeString(t) }
func (t *Struct) String() string    { return typeString(t) }
func (t *builtin) String() string   { return typeString(t) }

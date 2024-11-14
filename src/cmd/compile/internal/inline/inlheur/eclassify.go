// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package inlheur

import (
	"cmd/compile/internal/ir"
	"fmt"
	"os"
)

// ShouldFoldIfNameConstant analyzes expression tree 'e' to see
// whether it contains only combinations of simple references to all
// of the names in 'names' with selected constants + operators. The
// intent is to identify expression that could be folded away to a
// constant if the value of 'n' were available. Return value is TRUE
// if 'e' does look foldable given the value of 'n', and given that
// 'e' actually makes reference to 'n'. Some examples where the type
// of "n" is int64, type of "s" is string, and type of "p" is *byte:
//
//	Simple?		Expr
//	yes			n<10
//	yes			n*n-100
//	yes			(n < 10 || n > 100) && (n >= 12 || n <= 99 || n != 101)
//	yes			s == "foo"
//	yes			p == nil
//	no			n<foo()
//	no			n<1 || n>m
//	no			float32(n)<1.0
//	no			*p == 1
//	no			1 + 100
//	no			1 / n
//	no			1 + unsafe.Sizeof(n)
//
// To avoid complexities (e.g. nan, inf) we stay way from folding and
// floating point or complex operations (integers, bools, and strings
// only). We also try to be conservative about avoiding any operation
// that might result in a panic at runtime, e.g. for "n" with type
// int64:
//
//	1<<(n-9) < 100/(n<<9999)
//
// we would return FALSE due to the negative shift count and/or
// potential divide by zero.
func ShouldFoldIfNameConstant(n ir.Node, names []*ir.Name) bool {
	cl := makeExprClassifier(names)
	var doNode func(ir.Node) bool
	doNode = func { n ->
		ir.DoChildren(n, doNode)
		cl.Visit(n)
		return false
	}
	doNode(n)
	if cl.getdisp(n) != exprSimple {
		return false
	}
	for _, v := range cl.names {
		if !v {
			return false
		}
	}
	return true
}

// exprClassifier holds intermediate state about nodes within an
// expression tree being analyzed by ShouldFoldIfNameConstant. Here
// "name" is the name node passed in, and "disposition" stores the
// result of classifying a given IR node.
type exprClassifier struct {
	names       map[*ir.Name]bool
	disposition map[ir.Node]disp
}

type disp int

const (
	// no info on this expr
	exprNoInfo disp = iota

	// expr contains only literals
	exprLiterals

	// expr is legal combination of literals and specified names
	exprSimple
)

func (d disp) String() string {
	switch d {
	case exprNoInfo:
		return "noinfo"
	case exprSimple:
		return "simple"
	case exprLiterals:
		return "literals"
	default:
		return fmt.Sprintf("unknown<%d>", d)
	}
}

func makeExprClassifier(names []*ir.Name) *exprClassifier {
	m := make(map[*ir.Name]bool, len(names))
	for _, n := range names {
		m[n] = false
	}
	return &exprClassifier{
		names:       m,
		disposition: make(map[ir.Node]disp),
	}
}

// Visit sets the classification for 'n' based on the previously
// calculated classifications for n's children, as part of a bottom-up
// walk over an expression tree.
func (ec *exprClassifier) Visit(n ir.Node) {

	ndisp := exprNoInfo

	binparts := func(n ir.Node) (ir.Node, ir.Node) {
		if lex, ok := n.(*ir.LogicalExpr); ok {
			return lex.X, lex.Y
		} else if bex, ok := n.(*ir.BinaryExpr); ok {
			return bex.X, bex.Y
		} else {
			panic("bad")
		}
	}

	t := n.Type()
	if t == nil {
		if debugTrace&debugTraceExprClassify != 0 {
			fmt.Fprintf(os.Stderr, "=-= *** untyped op=%s\n",
				n.Op().String())
		}
	} else if t.IsInteger() || t.IsString() || t.IsBoolean() || t.HasNil() {
		switch n.Op() {
		// FIXME: maybe add support for OADDSTR?
		case ir.ONIL:
			ndisp = exprLiterals

		case ir.OLITERAL:
			if _, ok := n.(*ir.BasicLit); ok {
			} else {
				panic("unexpected")
			}
			ndisp = exprLiterals

		case ir.ONAME:
			nn := n.(*ir.Name)
			if _, ok := ec.names[nn]; ok {
				ndisp = exprSimple
				ec.names[nn] = true
			} else {
				sv := ir.StaticValue(n)
				if sv.Op() == ir.ONAME {
					nn = sv.(*ir.Name)
				}
				if _, ok := ec.names[nn]; ok {
					ndisp = exprSimple
					ec.names[nn] = true
				}
			}

		case ir.ONOT,
			ir.OPLUS,
			ir.ONEG:
			uex := n.(*ir.UnaryExpr)
			ndisp = ec.getdisp(uex.X)

		case ir.OEQ,
			ir.ONE,
			ir.OLT,
			ir.OGT,
			ir.OGE,
			ir.OLE:
			// compare ops
			x, y := binparts(n)
			ndisp = ec.dispmeet(x, y)
			if debugTrace&debugTraceExprClassify != 0 {
				fmt.Fprintf(os.Stderr, "=-= meet(%s,%s) = %s for op=%s\n",
					ec.getdisp(x), ec.getdisp(y), ec.dispmeet(x, y),
					n.Op().String())
			}
		case ir.OLSH,
			ir.ORSH,
			ir.ODIV,
			ir.OMOD:
			x, y := binparts(n)
			if ec.getdisp(y) == exprLiterals {
				ndisp = ec.dispmeet(x, y)
			}

		case ir.OADD,
			ir.OSUB,
			ir.OOR,
			ir.OXOR,
			ir.OMUL,
			ir.OAND,
			ir.OANDNOT,
			ir.OANDAND,
			ir.OOROR:
			x, y := binparts(n)
			if debugTrace&debugTraceExprClassify != 0 {
				fmt.Fprintf(os.Stderr, "=-= meet(%s,%s) = %s for op=%s\n",
					ec.getdisp(x), ec.getdisp(y), ec.dispmeet(x, y),
					n.Op().String())
			}
			ndisp = ec.dispmeet(x, y)
		}
	}

	if debugTrace&debugTraceExprClassify != 0 {
		fmt.Fprintf(os.Stderr, "=-= op=%s disp=%v\n", n.Op().String(),
			ndisp.String())
	}

	ec.disposition[n] = ndisp
}

func (ec *exprClassifier) getdisp(x ir.Node) disp {
	if d, ok := ec.disposition[x]; ok {
		return d
	} else {
		panic("missing node from disp table")
	}
}

// dispmeet performs a "meet" operation on the data flow states of
// node x and y (where the term "meet" is being drawn from traditional
// lattice-theoretical data flow analysis terminology).
func (ec *exprClassifier) dispmeet(x, y ir.Node) disp {
	xd := ec.getdisp(x)
	if xd == exprNoInfo {
		return exprNoInfo
	}
	yd := ec.getdisp(y)
	if yd == exprNoInfo {
		return exprNoInfo
	}
	if xd == exprSimple || yd == exprSimple {
		return exprSimple
	}
	if xd != exprLiterals || yd != exprLiterals {
		panic("unexpected")
	}
	return exprLiterals
}

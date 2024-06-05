// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ir

import (
	"bytes"
	"fmt"
	"go/constant"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strings"

	"unicode/utf8"

	"cmd/compile/internal/base"
	"cmd/compile/internal/types"
	"cmd/internal/src"
)

// Op

var OpNames = []string{
	OADDR:             "&",
	OADD:              "+",
	OADDSTR:           "+",
	OANDAND:           "&&",
	OANDNOT:           "&^",
	OAND:              "&",
	OAPPEND:           "append",
	OAS:               "=",
	OAS2:              "=",
	OBREAK:            "break",
	OCALL:             "function call", // not actual syntax
	OCAP:              "cap",
	OCASE:             "case",
	OCLEAR:            "clear",
	OCLOSE:            "close",
	OCOMPLEX:          "complex",
	OBITNOT:           "^",
	OCONTINUE:         "continue",
	OCOPY:             "copy",
	ODELETE:           "delete",
	ODEFER:            "defer",
	ODIV:              "/",
	OEQ:               "==",
	OFALL:             "fallthrough",
	OFOR:              "for",
	OGE:               ">=",
	OGOTO:             "goto",
	OGT:               ">",
	OIF:               "if",
	OIMAG:             "imag",
	OINLMARK:          "inlmark",
	ODEREF:            "*",
	OLEN:              "len",
	OLE:               "<=",
	OLSH:              "<<",
	OLT:               "<",
	OMAKE:             "make",
	ONEG:              "-",
	OMAX:              "max",
	OMIN:              "min",
	OMOD:              "%",
	OMUL:              "*",
	ONEW:              "new",
	ONE:               "!=",
	ONOT:              "!",
	OOROR:             "||",
	OOR:               "|",
	OPANIC:            "panic",
	OPLUS:             "+",
	OPRINTLN:          "println",
	OPRINT:            "print",
	ORANGE:            "range",
	OREAL:             "real",
	ORECV:             "<-",
	ORECOVER:          "recover",
	ORETURN:           "return",
	ORSH:              ">>",
	OSELECT:           "select",
	OSEND:             "<-",
	OSUB:              "-",
	OSWITCH:           "switch",
	OUNSAFEADD:        "unsafe.Add",
	OUNSAFESLICE:      "unsafe.Slice",
	OUNSAFESLICEDATA:  "unsafe.SliceData",
	OUNSAFESTRING:     "unsafe.String",
	OUNSAFESTRINGDATA: "unsafe.StringData",
	OXOR:              "^",
}

// GoString returns the Go syntax for the Op, or else its name.
func (o Op) GoString() string {
	if int(o) < len(OpNames) && OpNames[o] != "" {
		return OpNames[o]
	}
	return o.String()
}

// Format implements formatting for an Op.
// The valid formats are:
//
//	%v	Go syntax ("+", "<-", "print")
//	%+v	Debug syntax ("ADD", "RECV", "PRINT")
func (o Op) Format(s fmt.State, verb rune) {
	switch verb {
	default:
		fmt.Fprintf(s, "%%!%c(Op=%d)", verb, int(o))
	case 'v':
		if s.Flag('+') {
			// %+v is OMUL instead of "*"
			io.WriteString(s, o.String())
			return
		}
		io.WriteString(s, o.GoString())
	}
}

// Node

// fmtNode implements formatting for a Node n.
// Every Node implementation must define a Format method that calls fmtNode.
// The valid formats are:
//
//	%v	Go syntax
//	%L	Go syntax followed by " (type T)" if type is known.
//	%+v	Debug syntax, as in Dump.
func fmtNode(n Node, s fmt.State, verb rune) {
	// %+v prints Dump.
	// Otherwise we print Go syntax.
	if s.Flag('+') && verb == 'v' {
		dumpNode(s, n, 1)
		return
	}

	if verb != 'v' && verb != 'S' && verb != 'L' {
		fmt.Fprintf(s, "%%!%c(*Node=%p)", verb, n)
		return
	}

	if n == nil {
		fmt.Fprint(s, "<nil>")
		return
	}

	t := n.Type()
	if verb == 'L' && t != nil {
		if t.Kind() == types.TNIL {
			fmt.Fprint(s, "nil")
		} else if n.Op() == ONAME && n.Name().AutoTemp() {
			fmt.Fprintf(s, "%v value", t)
		} else {
			fmt.Fprintf(s, "%v (type %v)", n, t)
		}
		return
	}

	// TODO inlining produces expressions with ninits. we can't print these yet.

	if OpPrec[n.Op()] < 0 {
		stmtFmt(n, s)
		return
	}

	exprFmt(n, s, 0)
}

var OpPrec = []int{
	OAPPEND:           8,
	OBYTES2STR:        8,
	OARRAYLIT:         8,
	OSLICELIT:         8,
	ORUNES2STR:        8,
	OCALLFUNC:         8,
	OCALLINTER:        8,
	OCALLMETH:         8,
	OCALL:             8,
	OCAP:              8,
	OCLEAR:            8,
	OCLOSE:            8,
	OCOMPLIT:          8,
	OCONVIFACE:        8,
	OCONVNOP:          8,
	OCONV:             8,
	OCOPY:             8,
	ODELETE:           8,
	OGETG:             8,
	OLEN:              8,
	OLITERAL:          8,
	OMAKESLICE:        8,
	OMAKESLICECOPY:    8,
	OMAKE:             8,
	OMAPLIT:           8,
	OMAX:              8,
	OMIN:              8,
	ONAME:             8,
	ONEW:              8,
	ONIL:              8,
	ONONAME:           8,
	OPANIC:            8,
	OPAREN:            8,
	OPRINTLN:          8,
	OPRINT:            8,
	ORUNESTR:          8,
	OSLICE2ARR:        8,
	OSLICE2ARRPTR:     8,
	OSTR2BYTES:        8,
	OSTR2RUNES:        8,
	OSTRUCTLIT:        8,
	OTYPE:             8,
	OUNSAFEADD:        8,
	OUNSAFESLICE:      8,
	OUNSAFESLICEDATA:  8,
	OUNSAFESTRING:     8,
	OUNSAFESTRINGDATA: 8,
	OINDEXMAP:         8,
	OINDEX:            8,
	OSLICE:            8,
	OSLICESTR:         8,
	OSLICEARR:         8,
	OSLICE3:           8,
	OSLICE3ARR:        8,
	OSLICEHEADER:      8,
	OSTRINGHEADER:     8,
	ODOTINTER:         8,
	ODOTMETH:          8,
	ODOTPTR:           8,
	ODOTTYPE2:         8,
	ODOTTYPE:          8,
	ODOT:              8,
	OXDOT:             8,
	OMETHVALUE:        8,
	OMETHEXPR:         8,
	OPLUS:             7,
	ONOT:              7,
	OBITNOT:           7,
	ONEG:              7,
	OADDR:             7,
	ODEREF:            7,
	ORECV:             7,
	OMUL:              6,
	ODIV:              6,
	OMOD:              6,
	OLSH:              6,
	ORSH:              6,
	OAND:              6,
	OANDNOT:           6,
	OADD:              5,
	OSUB:              5,
	OOR:               5,
	OXOR:              5,
	OEQ:               4,
	OLT:               4,
	OLE:               4,
	OGE:               4,
	OGT:               4,
	ONE:               4,
	OSEND:             3,
	OANDAND:           2,
	OOROR:             1,

	// Statements handled by stmtfmt
	OAS:         -1,
	OAS2:        -1,
	OAS2DOTTYPE: -1,
	OAS2FUNC:    -1,
	OAS2MAPR:    -1,
	OAS2RECV:    -1,
	OASOP:       -1,
	OBLOCK:      -1,
	OBREAK:      -1,
	OCASE:       -1,
	OCONTINUE:   -1,
	ODCL:        -1,
	ODEFER:      -1,
	OFALL:       -1,
	OFOR:        -1,
	OGOTO:       -1,
	OIF:         -1,
	OLABEL:      -1,
	OGO:         -1,
	ORANGE:      -1,
	ORETURN:     -1,
	OSELECT:     -1,
	OSWITCH:     -1,

	OEND: 0,
}

// StmtWithInit reports whether op is a statement with an explicit init list.
func StmtWithInit(op Op) bool {
	switch op {
	case OIF, OFOR, OSWITCH:
		return true
	}
	return false
}

func stmtFmt(n Node, s fmt.State) {
	// NOTE(rsc): This code used to support the text-based
	// which was more aggressive about printing full Go syntax
	// (for example, an actual loop instead of "for loop").
	// The code is preserved for now in case we want to expand
	// any of those shortenings later. Or maybe we will delete
	// the code. But for now, keep it.
	const exportFormat = false

	// some statements allow for an init, but at most one,
	// but we may have an arbitrary number added, eg by typecheck
	// and inlining. If it doesn't fit the syntax, emit an enclosing
	// block starting with the init statements.

	// if we can just say "for" n->ninit; ... then do so
	simpleinit := len(n.Init()) == 1 && len(n.Init()[0].Init()) == 0 && StmtWithInit(n.Op())

	// otherwise, print the inits as separate statements
	complexinit := len(n.Init()) != 0 && !simpleinit && exportFormat

	// but if it was for if/for/switch, put in an extra surrounding block to limit the scope
	extrablock := complexinit && StmtWithInit(n.Op())

	if extrablock {
		fmt.Fprint(s, "{")
	}

	if complexinit {
		fmt.Fprintf(s, " %v; ", n.Init())
	}

	switch n.Op() {
	case ODCL:
		n := n.(*Decl)
		fmt.Fprintf(s, "var %v %v", n.X.Sym(), n.X.Type())

	// Don't export "v = <N>" initializing statements, hope they're always
	// preceded by the DCL which will be re-parsed and typechecked to reproduce
	// the "v = <N>" again.
	case OAS:
		n := n.(*AssignStmt)
		if n.Def && !complexinit {
			fmt.Fprintf(s, "%v := %v", n.X, n.Y)
		} else {
			fmt.Fprintf(s, "%v = %v", n.X, n.Y)
		}

	case OASOP:
		n := n.(*AssignOpStmt)
		if n.IncDec {
			if n.AsOp == OADD {
				fmt.Fprintf(s, "%v++", n.X)
			} else {
				fmt.Fprintf(s, "%v--", n.X)
			}
			break
		}

		fmt.Fprintf(s, "%v %v= %v", n.X, n.AsOp, n.Y)

	case OAS2, OAS2DOTTYPE, OAS2FUNC, OAS2MAPR, OAS2RECV:
		n := n.(*AssignListStmt)
		if n.Def && !complexinit {
			fmt.Fprintf(s, "%.v := %.v", n.Lhs, n.Rhs)
		} else {
			fmt.Fprintf(s, "%.v = %.v", n.Lhs, n.Rhs)
		}

	case OBLOCK:
		n := n.(*BlockStmt)
		if len(n.List) != 0 {
			fmt.Fprintf(s, "%v", n.List)
		}

	case ORETURN:
		n := n.(*ReturnStmt)
		fmt.Fprintf(s, "return %.v", n.Results)

	case OTAILCALL:
		n := n.(*TailCallStmt)
		fmt.Fprintf(s, "tailcall %v", n.Call)

	case OINLMARK:
		n := n.(*InlineMarkStmt)
		fmt.Fprintf(s, "inlmark %d", n.Index)

	case OGO:
		n := n.(*GoDeferStmt)
		fmt.Fprintf(s, "go %v", n.Call)

	case ODEFER:
		n := n.(*GoDeferStmt)
		fmt.Fprintf(s, "defer %v", n.Call)

	case OIF:
		n := n.(*IfStmt)
		if simpleinit {
			fmt.Fprintf(s, "if %v; %v { %v }", n.Init()[0], n.Cond, n.Body)
		} else {
			fmt.Fprintf(s, "if %v { %v }", n.Cond, n.Body)
		}
		if len(n.Else) != 0 {
			fmt.Fprintf(s, " else { %v }", n.Else)
		}

	case OFOR:
		n := n.(*ForStmt)
		if !exportFormat { // TODO maybe only if FmtShort, same below
			fmt.Fprintf(s, "for loop")
			break
		}

		fmt.Fprint(s, "for")
		if n.DistinctVars {
			fmt.Fprint(s, " /* distinct */")
		}
		if simpleinit {
			fmt.Fprintf(s, " %v;", n.Init()[0])
		} else if n.Post != nil {
			fmt.Fprint(s, " ;")
		}

		if n.Cond != nil {
			fmt.Fprintf(s, " %v", n.Cond)
		}

		if n.Post != nil {
			fmt.Fprintf(s, "; %v", n.Post)
		} else if simpleinit {
			fmt.Fprint(s, ";")
		}

		fmt.Fprintf(s, " { %v }", n.Body)

	case ORANGE:
		n := n.(*RangeStmt)
		if !exportFormat {
			fmt.Fprint(s, "for loop")
			break
		}

		fmt.Fprint(s, "for")
		if n.Key != nil {
			fmt.Fprintf(s, " %v", n.Key)
			if n.Value != nil {
				fmt.Fprintf(s, ", %v", n.Value)
			}
			fmt.Fprint(s, " =")
		}
		fmt.Fprintf(s, " range %v { %v }", n.X, n.Body)
		if n.DistinctVars {
			fmt.Fprint(s, " /* distinct vars */")
		}

	case OSELECT:
		n := n.(*SelectStmt)
		if !exportFormat {
			fmt.Fprintf(s, "%v statement", n.Op())
			break
		}
		fmt.Fprintf(s, "select { %v }", n.Cases)

	case OSWITCH:
		n := n.(*SwitchStmt)
		if !exportFormat {
			fmt.Fprintf(s, "%v statement", n.Op())
			break
		}
		fmt.Fprintf(s, "switch")
		if simpleinit {
			fmt.Fprintf(s, " %v;", n.Init()[0])
		}
		if n.Tag != nil {
			fmt.Fprintf(s, " %v ", n.Tag)
		}
		fmt.Fprintf(s, " { %v }", n.Cases)

	case OCASE:
		n := n.(*CaseClause)
		if len(n.List) != 0 {
			fmt.Fprintf(s, "case %.v", n.List)
		} else {
			fmt.Fprint(s, "default")
		}
		fmt.Fprintf(s, ": %v", n.Body)

	case OBREAK, OCONTINUE, OGOTO, OFALL:
		n := n.(*BranchStmt)
		if n.Label != nil {
			fmt.Fprintf(s, "%v %v", n.Op(), n.Label)
		} else {
			fmt.Fprintf(s, "%v", n.Op())
		}

	case OLABEL:
		n := n.(*LabelStmt)
		fmt.Fprintf(s, "%v: ", n.Label)
	}

	if extrablock {
		fmt.Fprint(s, "}")
	}
}

func exprFmt(n Node, s fmt.State, prec int) {
	// NOTE(rsc): This code used to support the text-based
	// which was more aggressive about printing full Go syntax
	// (for example, an actual loop instead of "for loop").
	// The code is preserved for now in case we want to expand
	// any of those shortenings later. Or maybe we will delete
	// the code. But for now, keep it.
	const exportFormat = false

	for {
		if n == nil {
			fmt.Fprint(s, "<nil>")
			return
		}

		// Skip implicit operations introduced during typechecking.
		switch nn := n; nn.Op() {
		case OADDR:
			nn := nn.(*AddrExpr)
			if nn.Implicit() {
				n = nn.X
				continue
			}
		case ODEREF:
			nn := nn.(*StarExpr)
			if nn.Implicit() {
				n = nn.X
				continue
			}
		case OCONV, OCONVNOP, OCONVIFACE:
			nn := nn.(*ConvExpr)
			if nn.Implicit() {
				n = nn.X
				continue
			}
		}

		break
	}

	nprec := OpPrec[n.Op()]
	if n.Op() == OTYPE && n.Type() != nil && n.Type().IsPtr() {
		nprec = OpPrec[ODEREF]
	}

	if prec > nprec {
		fmt.Fprintf(s, "(%v)", n)
		return
	}

	switch n.Op() {
	case OPAREN:
		n := n.(*ParenExpr)
		fmt.Fprintf(s, "(%v)", n.X)

	case ONIL:
		fmt.Fprint(s, "nil")

	case OLITERAL:
		if n.Sym() != nil {
			fmt.Fprint(s, n.Sym())
			return
		}

		typ := n.Type()
		val := n.Val()

		// Special case for rune constants.
		if typ == types.RuneType || typ == types.UntypedRune {
			if x, ok := constant.Uint64Val(val); ok && x <= utf8.MaxRune {
				fmt.Fprintf(s, "%q", x)
				return
			}
		}

		// Only include typ if it's neither the default nor untyped type
		// for the constant value.
		if k := val.Kind(); typ == types.Types[types.DefaultKinds[k]] || typ == types.UntypedTypes[k] {
			fmt.Fprint(s, val)
		} else {
			fmt.Fprintf(s, "%v(%v)", typ, val)
		}

	case ODCLFUNC:
		n := n.(*Func)
		if sym := n.Sym(); sym != nil {
			fmt.Fprint(s, sym)
			return
		}
		fmt.Fprintf(s, "<unnamed Func>")

	case ONAME:
		n := n.(*Name)
		// Special case: name used as local variable in export.
		// _ becomes ~b%d internally; print as _ for export
		if !exportFormat && n.Sym() != nil && n.Sym().Name[0] == '~' && n.Sym().Name[1] == 'b' {
			fmt.Fprint(s, "_")
			return
		}
		fallthrough
	case ONONAME:
		fmt.Fprint(s, n.Sym())

	case OLINKSYMOFFSET:
		n := n.(*LinksymOffsetExpr)
		fmt.Fprintf(s, "(%v)(%s@%d)", n.Type(), n.Linksym.Name, n.Offset_)

	case OTYPE:
		if n.Type() == nil && n.Sym() != nil {
			fmt.Fprint(s, n.Sym())
			return
		}
		fmt.Fprintf(s, "%v", n.Type())

	case OCLOSURE:
		n := n.(*ClosureExpr)
		if !exportFormat {
			fmt.Fprint(s, "func literal")
			return
		}
		fmt.Fprintf(s, "%v { %v }", n.Type(), n.Func.Body)

	case OPTRLIT:
		n := n.(*AddrExpr)
		fmt.Fprintf(s, "&%v", n.X)

	case OCOMPLIT, OSTRUCTLIT, OARRAYLIT, OSLICELIT, OMAPLIT:
		n := n.(*CompLitExpr)
		if n.Implicit() {
			fmt.Fprintf(s, "... argument")
			return
		}
		fmt.Fprintf(s, "%v{%s}", n.Type(), ellipsisIf(len(n.List) != 0))

	case OKEY:
		n := n.(*KeyExpr)
		if n.Key != nil && n.Value != nil {
			fmt.Fprintf(s, "%v:%v", n.Key, n.Value)
			return
		}

		if n.Key == nil && n.Value != nil {
			fmt.Fprintf(s, ":%v", n.Value)
			return
		}
		if n.Key != nil && n.Value == nil {
			fmt.Fprintf(s, "%v:", n.Key)
			return
		}
		fmt.Fprint(s, ":")

	case OSTRUCTKEY:
		n := n.(*StructKeyExpr)
		fmt.Fprintf(s, "%v:%v", n.Field, n.Value)

	case OXDOT, ODOT, ODOTPTR, ODOTINTER, ODOTMETH, OMETHVALUE, OMETHEXPR:
		n := n.(*SelectorExpr)
		exprFmt(n.X, s, nprec)
		if n.Sel == nil {
			fmt.Fprint(s, ".<nil>")
			return
		}
		fmt.Fprintf(s, ".%s", n.Sel.Name)

	case ODOTTYPE, ODOTTYPE2:
		n := n.(*TypeAssertExpr)
		exprFmt(n.X, s, nprec)
		fmt.Fprintf(s, ".(%v)", n.Type())

	case OINDEX, OINDEXMAP:
		n := n.(*IndexExpr)
		exprFmt(n.X, s, nprec)
		fmt.Fprintf(s, "[%v]", n.Index)

	case OSLICE, OSLICESTR, OSLICEARR, OSLICE3, OSLICE3ARR:
		n := n.(*SliceExpr)
		exprFmt(n.X, s, nprec)
		fmt.Fprint(s, "[")
		if n.Low != nil {
			fmt.Fprint(s, n.Low)
		}
		fmt.Fprint(s, ":")
		if n.High != nil {
			fmt.Fprint(s, n.High)
		}
		if n.Op().IsSlice3() {
			fmt.Fprint(s, ":")
			if n.Max != nil {
				fmt.Fprint(s, n.Max)
			}
		}
		fmt.Fprint(s, "]")

	case OSLICEHEADER:
		n := n.(*SliceHeaderExpr)
		fmt.Fprintf(s, "sliceheader{%v,%v,%v}", n.Ptr, n.Len, n.Cap)

	case OCOMPLEX, OCOPY, OUNSAFEADD, OUNSAFESLICE:
		n := n.(*BinaryExpr)
		fmt.Fprintf(s, "%v(%v, %v)", n.Op(), n.X, n.Y)

	case OCONV,
		OCONVIFACE,
		OCONVNOP,
		OBYTES2STR,
		ORUNES2STR,
		OSTR2BYTES,
		OSTR2RUNES,
		ORUNESTR,
		OSLICE2ARR,
		OSLICE2ARRPTR:
		n := n.(*ConvExpr)
		if n.Type() == nil || n.Type().Sym() == nil {
			fmt.Fprintf(s, "(%v)", n.Type())
		} else {
			fmt.Fprintf(s, "%v", n.Type())
		}
		fmt.Fprintf(s, "(%v)", n.X)

	case OREAL,
		OIMAG,
		OCAP,
		OCLEAR,
		OCLOSE,
		OLEN,
		ONEW,
		OPANIC:
		n := n.(*UnaryExpr)
		fmt.Fprintf(s, "%v(%v)", n.Op(), n.X)

	case OAPPEND,
		ODELETE,
		OMAKE,
		OMAX,
		OMIN,
		ORECOVER,
		OPRINT,
		OPRINTLN:
		n := n.(*CallExpr)
		if n.IsDDD {
			fmt.Fprintf(s, "%v(%.v...)", n.Op(), n.Args)
			return
		}
		fmt.Fprintf(s, "%v(%.v)", n.Op(), n.Args)

	case OCALL, OCALLFUNC, OCALLINTER, OCALLMETH, OGETG:
		n := n.(*CallExpr)
		exprFmt(n.Fun, s, nprec)
		if n.IsDDD {
			fmt.Fprintf(s, "(%.v...)", n.Args)
			return
		}
		fmt.Fprintf(s, "(%.v)", n.Args)

	case OINLCALL:
		n := n.(*InlinedCallExpr)
		// TODO(mdempsky): Print Init and/or Body?
		if len(n.ReturnVars) == 1 {
			fmt.Fprintf(s, "%v", n.ReturnVars[0])
			return
		}
		fmt.Fprintf(s, "(.%v)", n.ReturnVars)

	case OMAKEMAP, OMAKECHAN, OMAKESLICE:
		n := n.(*MakeExpr)
		if n.Cap != nil {
			fmt.Fprintf(s, "make(%v, %v, %v)", n.Type(), n.Len, n.Cap)
			return
		}
		if n.Len != nil && (n.Op() == OMAKESLICE || !n.Len.Type().IsUntyped()) {
			fmt.Fprintf(s, "make(%v, %v)", n.Type(), n.Len)
			return
		}
		fmt.Fprintf(s, "make(%v)", n.Type())

	case OMAKESLICECOPY:
		n := n.(*MakeExpr)
		fmt.Fprintf(s, "makeslicecopy(%v, %v, %v)", n.Type(), n.Len, n.Cap)

	case OPLUS, ONEG, OBITNOT, ONOT, ORECV:
		// Unary
		n := n.(*UnaryExpr)
		fmt.Fprintf(s, "%v", n.Op())
		if n.X != nil && n.X.Op() == n.Op() {
			fmt.Fprint(s, " ")
		}
		exprFmt(n.X, s, nprec+1)

	case OADDR:
		n := n.(*AddrExpr)
		fmt.Fprintf(s, "%v", n.Op())
		if n.X != nil && n.X.Op() == n.Op() {
			fmt.Fprint(s, " ")
		}
		exprFmt(n.X, s, nprec+1)

	case ODEREF:
		n := n.(*StarExpr)
		fmt.Fprintf(s, "%v", n.Op())
		exprFmt(n.X, s, nprec+1)

		// Binary
	case OADD,
		OAND,
		OANDNOT,
		ODIV,
		OEQ,
		OGE,
		OGT,
		OLE,
		OLT,
		OLSH,
		OMOD,
		OMUL,
		ONE,
		OOR,
		ORSH,
		OSUB,
		OXOR:
		n := n.(*BinaryExpr)
		exprFmt(n.X, s, nprec)
		fmt.Fprintf(s, " %v ", n.Op())
		exprFmt(n.Y, s, nprec+1)

	case OANDAND,
		OOROR:
		n := n.(*LogicalExpr)
		exprFmt(n.X, s, nprec)
		fmt.Fprintf(s, " %v ", n.Op())
		exprFmt(n.Y, s, nprec+1)

	case OSEND:
		n := n.(*SendStmt)
		exprFmt(n.Chan, s, nprec)
		fmt.Fprintf(s, " <- ")
		exprFmt(n.Value, s, nprec+1)

	case OADDSTR:
		n := n.(*AddStringExpr)
		for i, n1 := range n.List {
			if i != 0 {
				fmt.Fprint(s, " + ")
			}
			exprFmt(n1, s, nprec)
		}
	default:
		fmt.Fprintf(s, "<node %v>", n.Op())
	}
}

func ellipsisIf(b bool) string {
	if b {
		return "..."
	}
	return ""
}

// Nodes

// Format implements formatting for a Nodes.
// The valid formats are:
//
//	%v	Go syntax, semicolon-separated
//	%.v	Go syntax, comma-separated
//	%+v	Debug syntax, as in DumpList.
func (l Nodes) Format(s fmt.State, verb rune) {
	if s.Flag('+') && verb == 'v' {
		// %+v is DumpList output
		dumpNodes(s, l, 1)
		return
	}

	if verb != 'v' {
		fmt.Fprintf(s, "%%!%c(Nodes)", verb)
		return
	}

	sep := "; "
	if _, ok := s.Precision(); ok { // %.v is expr list
		sep = ", "
	}

	for i, n := range l {
		fmt.Fprint(s, n)
		if i+1 < len(l) {
			fmt.Fprint(s, sep)
		}
	}
}

// Dump

// Dump prints the message s followed by a debug dump of n.
func Dump(s string, n Node) {
	fmt.Printf("%s%+v\n", s, n)
}

// DumpList prints the message s followed by a debug dump of each node in the list.
func DumpList(s string, list Nodes) {
	var buf bytes.Buffer
	FDumpList(&buf, s, list)
	os.Stdout.Write(buf.Bytes())
}

// FDumpList prints to w the message s followed by a debug dump of each node in the list.
func FDumpList(w io.Writer, s string, list Nodes) {
	io.WriteString(w, s)
	dumpNodes(w, list, 1)
	io.WriteString(w, "\n")
}

// indent prints indentation to w.
func indent(w io.Writer, depth int) {
	fmt.Fprint(w, "\n")
	for i := 0; i < depth; i++ {
		fmt.Fprint(w, ".   ")
	}
}

// EscFmt is set by the escape analysis code to add escape analysis details to the node print.
var EscFmt func(n Node) string

// dumpNodeHeader prints the debug-format node header line to w.
func dumpNodeHeader(w io.Writer, n Node) {
	// Useful to see which nodes in an AST printout are actually identical
	if base.Debug.DumpPtrs != 0 {
		fmt.Fprintf(w, " p(%p)", n)
	}

	if base.Debug.DumpPtrs != 0 && n.Name() != nil && n.Name().Defn != nil {
		// Useful to see where Defn is set and what node it points to
		fmt.Fprintf(w, " defn(%p)", n.Name().Defn)
	}

	if base.Debug.DumpPtrs != 0 && n.Name() != nil && n.Name().Curfn != nil {
		// Useful to see where Defn is set and what node it points to
		fmt.Fprintf(w, " curfn(%p)", n.Name().Curfn)
	}
	if base.Debug.DumpPtrs != 0 && n.Name() != nil && n.Name().Outer != nil {
		// Useful to see where Defn is set and what node it points to
		fmt.Fprintf(w, " outer(%p)", n.Name().Outer)
	}

	if EscFmt != nil {
		if esc := EscFmt(n); esc != "" {
			fmt.Fprintf(w, " %s", esc)
		}
	}

	if n.Sym() != nil && n.Op() != ONAME && n.Op() != ONONAME && n.Op() != OTYPE {
		fmt.Fprintf(w, " %+v", n.Sym())
	}

	// Print Node-specific fields of basic type in header line.
	v := reflect.ValueOf(n).Elem()
	t := v.Type()
	nf := t.NumField()
	for i := 0; i < nf; i++ {
		tf := t.Field(i)
		if tf.PkgPath != "" {
			// skip unexported field - Interface will fail
			continue
		}
		k := tf.Type.Kind()
		if reflect.Bool <= k && k <= reflect.Complex128 {
			name := strings.TrimSuffix(tf.Name, "_")
			vf := v.Field(i)
			vfi := vf.Interface()
			if name == "Offset" && vfi == types.BADWIDTH || name != "Offset" && vf.IsZero() {
				continue
			}
			if vfi == true {
				fmt.Fprintf(w, " %s", name)
			} else {
				fmt.Fprintf(w, " %s:%+v", name, vf.Interface())
			}
		}
	}

	// Print Node-specific booleans by looking for methods.
	// Different v, t from above - want *Struct not Struct, for methods.
	v = reflect.ValueOf(n)
	t = v.Type()
	nm := t.NumMethod()
	for i := 0; i < nm; i++ {
		tm := t.Method(i)
		if tm.PkgPath != "" {
			// skip unexported method - call will fail
			continue
		}
		m := v.Method(i)
		mt := m.Type()
		if mt.NumIn() == 0 && mt.NumOut() == 1 && mt.Out(0).Kind() == reflect.Bool {
			// TODO(rsc): Remove the func/defer/recover wrapping,
			// which is guarding against panics in miniExpr,
			// once we get down to the simpler state in which
			// nodes have no getter methods that aren't allowed to be called.
			func() {
				defer func() { recover() }()
				if m.Call(nil)[0].Bool() {
					name := strings.TrimSuffix(tm.Name, "_")
					fmt.Fprintf(w, " %s", name)
				}
			}()
		}
	}

	if n.Op() == OCLOSURE {
		n := n.(*ClosureExpr)
		if fn := n.Func; fn != nil && fn.Nname.Sym() != nil {
			fmt.Fprintf(w, " fnName(%+v)", fn.Nname.Sym())
		}
	}

	if n.Type() != nil {
		if n.Op() == OTYPE {
			fmt.Fprintf(w, " type")
		}
		fmt.Fprintf(w, " %+v", n.Type())
	}
	if n.Typecheck() != 0 {
		fmt.Fprintf(w, " tc(%d)", n.Typecheck())
	}

	if n.Pos().IsKnown() {
		fmt.Fprint(w, " # ")
		switch n.Pos().IsStmt() {
		case src.PosNotStmt:
			fmt.Fprint(w, "_") // "-" would be confusing
		case src.PosIsStmt:
			fmt.Fprint(w, "+")
		}
		sep := ""
		base.Ctxt.AllPos(n.Pos(), func(pos src.Pos) {
			fmt.Fprint(w, sep)
			sep = " "
			// TODO(mdempsky): Print line pragma details too.
			file := filepath.Base(pos.Filename())
			// Note: this output will be parsed by ssa/html.go:(*HTMLWriter).WriteAST. Keep in sync.
			fmt.Fprintf(w, "%s:%d:%d", file, pos.Line(), pos.Col())
		})
	}
}

func dumpNode(w io.Writer, n Node, depth int) {
	indent(w, depth)
	if depth > 40 {
		fmt.Fprint(w, "...")
		return
	}

	if n == nil {
		fmt.Fprint(w, "NilIrNode")
		return
	}

	if len(n.Init()) != 0 {
		fmt.Fprintf(w, "%+v-init", n.Op())
		dumpNodes(w, n.Init(), depth+1)
		indent(w, depth)
	}

	switch n.Op() {
	default:
		fmt.Fprintf(w, "%+v", n.Op())
		dumpNodeHeader(w, n)

	case OLITERAL:
		fmt.Fprintf(w, "%+v-%v", n.Op(), n.Val())
		dumpNodeHeader(w, n)
		return

	case ONAME, ONONAME:
		if n.Sym() != nil {
			fmt.Fprintf(w, "%+v-%+v", n.Op(), n.Sym())
		} else {
			fmt.Fprintf(w, "%+v", n.Op())
		}
		dumpNodeHeader(w, n)
		return

	case OLINKSYMOFFSET:
		n := n.(*LinksymOffsetExpr)
		fmt.Fprintf(w, "%+v-%v", n.Op(), n.Linksym)
		// Offset is almost always 0, so only print when it's interesting.
		if n.Offset_ != 0 {
			fmt.Fprintf(w, "%+v", n.Offset_)
		}
		dumpNodeHeader(w, n)

	case OASOP:
		n := n.(*AssignOpStmt)
		fmt.Fprintf(w, "%+v-%+v", n.Op(), n.AsOp)
		dumpNodeHeader(w, n)

	case OTYPE:
		fmt.Fprintf(w, "%+v %+v", n.Op(), n.Sym())
		dumpNodeHeader(w, n)
		return

	case OCLOSURE:
		fmt.Fprintf(w, "%+v", n.Op())
		dumpNodeHeader(w, n)

	case ODCLFUNC:
		// Func has many fields we don't want to print.
		// Bypass reflection and just print what we want.
		n := n.(*Func)
		fmt.Fprintf(w, "%+v", n.Op())
		dumpNodeHeader(w, n)
		fn := n
		if len(fn.Dcl) > 0 {
			indent(w, depth)
			fmt.Fprintf(w, "%+v-Dcl", n.Op())
			for _, dcl := range n.Dcl {
				dumpNode(w, dcl, depth+1)
			}
		}
		if len(fn.ClosureVars) > 0 {
			indent(w, depth)
			fmt.Fprintf(w, "%+v-ClosureVars", n.Op())
			for _, cv := range fn.ClosureVars {
				dumpNode(w, cv, depth+1)
			}
		}
		if len(fn.Body) > 0 {
			indent(w, depth)
			fmt.Fprintf(w, "%+v-body", n.Op())
			dumpNodes(w, fn.Body, depth+1)
		}
		return
	}

	v := reflect.ValueOf(n).Elem()
	t := reflect.TypeOf(n).Elem()
	nf := t.NumField()
	for i := 0; i < nf; i++ {
		tf := t.Field(i)
		vf := v.Field(i)
		if tf.PkgPath != "" {
			// skip unexported field - Interface will fail
			continue
		}
		switch tf.Type.Kind() {
		case reflect.Interface, reflect.Ptr, reflect.Slice:
			if vf.IsNil() {
				continue
			}
		}
		name := strings.TrimSuffix(tf.Name, "_")
		// Do not bother with field name header lines for the
		// most common positional arguments: unary, binary expr,
		// index expr, send stmt, go and defer call expression.
		switch name {
		case "X", "Y", "Index", "Chan", "Value", "Call":
			name = ""
		}
		switch val := vf.Interface().(type) {
		case Node:
			if name != "" {
				indent(w, depth)
				fmt.Fprintf(w, "%+v-%s", n.Op(), name)
			}
			dumpNode(w, val, depth+1)
		case Nodes:
			if len(val) == 0 {
				continue
			}
			if name != "" {
				indent(w, depth)
				fmt.Fprintf(w, "%+v-%s", n.Op(), name)
			}
			dumpNodes(w, val, depth+1)
		default:
			if vf.Kind() == reflect.Slice && vf.Type().Elem().Implements(nodeType) {
				if vf.Len() == 0 {
					continue
				}
				if name != "" {
					indent(w, depth)
					fmt.Fprintf(w, "%+v-%s", n.Op(), name)
				}
				for i, n := 0, vf.Len(); i < n; i++ {
					dumpNode(w, vf.Index(i).Interface().(Node), depth+1)
				}
			}
		}
	}
}

var nodeType = reflect.TypeOf((*Node)(nil)).Elem()

func dumpNodes(w io.Writer, list Nodes, depth int) {
	if len(list) == 0 {
		fmt.Fprintf(w, " <nil>")
		return
	}

	for _, n := range list {
		dumpNode(w, n, depth)
	}
}

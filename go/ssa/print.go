// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// This file implements the String() methods for all Value and
// Instruction types.

import (
	"bytes"
	"fmt"
	"go/types"
	"io"
	"reflect"
	"sort"
	"strings"

	"golang.org/x/tools/go/types/typeutil"
	"golang.org/x/tools/internal/typeparams"
)

// relName returns the name of v relative to i.
// In most cases, this is identical to v.Name(), but references to
// Functions (including methods) and Globals use RelString and
// all types are displayed with relType, so that only cross-package
// references are package-qualified.
func relName(v Value, i Instruction) string {
	var from *types.Package
	if i != nil {
		from = i.Parent().relPkg()
	}
	switch v := v.(type) {
	case Member: // *Function or *Global
		return v.RelString(from)
	case *Const:
		return v.RelString(from)
	}
	return v.Name()
}

// normalizeAnyFortesting controls whether we replace occurrences of
// interface{} with any. It is only used for normalizing test output.
var normalizeAnyForTesting bool

func relType(t types.Type, from *types.Package) string {
	s := types.TypeString(t, types.RelativeTo(from))
	if normalizeAnyForTesting {
		s = strings.ReplaceAll(s, "interface{}", "any")
	}
	return s
}

func relString(m Member, from *types.Package) string {
	// NB: not all globals have an Object (e.g. init$guard),
	// so use Package().Object not Object.Package().
	if pkg := m.Package().Pkg; pkg != nil && pkg != from {
		return fmt.Sprintf("%s.%s", pkg.Path(), m.Name())
	}
	return m.Name()
}

// Value.String()
//
// This method is provided only for debugging.
// It never appears in disassembly, which uses Value.Name().

func (v *Parameter) String() string {
	from := v.Parent().relPkg()
	return fmt.Sprintf("parameter %s : %s", v.Name(), relType(v.Type(), from))
}

func (v *FreeVar) String() string {
	from := v.Parent().relPkg()
	return fmt.Sprintf("freevar %s : %s", v.Name(), relType(v.Type(), from))
}

func (v *Builtin) String() string {
	return fmt.Sprintf("builtin %s", v.Name())
}

// Instruction.String()

func (v *Alloc) String() string {
	op := "local"
	if v.Heap {
		op = "new"
	}
	from := v.Parent().relPkg()
	return fmt.Sprintf("%s %s (%s)", op, relType(deref(v.Type()), from), v.Comment)
}

func (v *Phi) String() string {
	var b bytes.Buffer
	b.WriteString("phi [")
	for i, edge := range v.Edges {
		if i > 0 {
			b.WriteString(", ")
		}
		// Be robust against malformed CFG.
		if v.block == nil {
			b.WriteString("??")
			continue
		}
		block := -1
		if i < len(v.block.Preds) {
			block = v.block.Preds[i].Index
		}
		fmt.Fprintf(&b, "%d: ", block)
		edgeVal := "<nil>" // be robust
		if edge != nil {
			edgeVal = relName(edge, v)
		}
		b.WriteString(edgeVal)
	}
	b.WriteString("]")
	if v.Comment != "" {
		b.WriteString(" #")
		b.WriteString(v.Comment)
	}
	return b.String()
}

func printCall(v *CallCommon, prefix string, instr Instruction) string {
	var b bytes.Buffer
	b.WriteString(prefix)
	if !v.IsInvoke() {
		b.WriteString(relName(v.Value, instr))
	} else {
		fmt.Fprintf(&b, "invoke %s.%s", relName(v.Value, instr), v.Method.Name())
	}
	b.WriteString("(")
	for i, arg := range v.Args {
		if i > 0 {
			b.WriteString(", ")
		}
		b.WriteString(relName(arg, instr))
	}
	if v.Signature().Variadic() {
		b.WriteString("...")
	}
	b.WriteString(")")
	return b.String()
}

func (c *CallCommon) String() string {
	return printCall(c, "", nil)
}

func (v *Call) String() string {
	return printCall(&v.Call, "", v)
}

func (v *BinOp) String() string {
	return fmt.Sprintf("%s %s %s", relName(v.X, v), v.Op.String(), relName(v.Y, v))
}

func (v *UnOp) String() string {
	return fmt.Sprintf("%s%s%s", v.Op, relName(v.X, v), commaOk(v.CommaOk))
}

func printConv(prefix string, v, x Value) string {
	from := v.Parent().relPkg()
	return fmt.Sprintf("%s %s <- %s (%s)",
		prefix,
		relType(v.Type(), from),
		relType(x.Type(), from),
		relName(x, v.(Instruction)))
}

func (v *ChangeType) String() string          { return printConv("changetype", v, v.X) }
func (v *Convert) String() string             { return printConv("convert", v, v.X) }
func (v *ChangeInterface) String() string     { return printConv("change interface", v, v.X) }
func (v *SliceToArrayPointer) String() string { return printConv("slice to array pointer", v, v.X) }
func (v *MakeInterface) String() string       { return printConv("make", v, v.X) }

func (v *MakeClosure) String() string {
	var b bytes.Buffer
	fmt.Fprintf(&b, "make closure %s", relName(v.Fn, v))
	if v.Bindings != nil {
		b.WriteString(" [")
		for i, c := range v.Bindings {
			if i > 0 {
				b.WriteString(", ")
			}
			b.WriteString(relName(c, v))
		}
		b.WriteString("]")
	}
	return b.String()
}

func (v *MakeSlice) String() string {
	from := v.Parent().relPkg()
	return fmt.Sprintf("make %s %s %s",
		relType(v.Type(), from),
		relName(v.Len, v),
		relName(v.Cap, v))
}

func (v *Slice) String() string {
	var b bytes.Buffer
	b.WriteString("slice ")
	b.WriteString(relName(v.X, v))
	b.WriteString("[")
	if v.Low != nil {
		b.WriteString(relName(v.Low, v))
	}
	b.WriteString(":")
	if v.High != nil {
		b.WriteString(relName(v.High, v))
	}
	if v.Max != nil {
		b.WriteString(":")
		b.WriteString(relName(v.Max, v))
	}
	b.WriteString("]")
	return b.String()
}

func (v *MakeMap) String() string {
	res := ""
	if v.Reserve != nil {
		res = relName(v.Reserve, v)
	}
	from := v.Parent().relPkg()
	return fmt.Sprintf("make %s %s", relType(v.Type(), from), res)
}

func (v *MakeChan) String() string {
	from := v.Parent().relPkg()
	return fmt.Sprintf("make %s %s", relType(v.Type(), from), relName(v.Size, v))
}

func (v *FieldAddr) String() string {
	st := typeparams.CoreType(deref(v.X.Type())).(*types.Struct)
	// Be robust against a bad index.
	name := "?"
	if 0 <= v.Field && v.Field < st.NumFields() {
		name = st.Field(v.Field).Name()
	}
	return fmt.Sprintf("&%s.%s [#%d]", relName(v.X, v), name, v.Field)
}

func (v *Field) String() string {
	st := typeparams.CoreType(v.X.Type()).(*types.Struct)
	// Be robust against a bad index.
	name := "?"
	if 0 <= v.Field && v.Field < st.NumFields() {
		name = st.Field(v.Field).Name()
	}
	return fmt.Sprintf("%s.%s [#%d]", relName(v.X, v), name, v.Field)
}

func (v *IndexAddr) String() string {
	return fmt.Sprintf("&%s[%s]", relName(v.X, v), relName(v.Index, v))
}

func (v *Index) String() string {
	return fmt.Sprintf("%s[%s]", relName(v.X, v), relName(v.Index, v))
}

func (v *Lookup) String() string {
	return fmt.Sprintf("%s[%s]%s", relName(v.X, v), relName(v.Index, v), commaOk(v.CommaOk))
}

func (v *Range) String() string {
	return "range " + relName(v.X, v)
}

func (v *Next) String() string {
	return "next " + relName(v.Iter, v)
}

func (v *TypeAssert) String() string {
	from := v.Parent().relPkg()
	return fmt.Sprintf("typeassert%s %s.(%s)", commaOk(v.CommaOk), relName(v.X, v), relType(v.AssertedType, from))
}

func (v *Extract) String() string {
	return fmt.Sprintf("extract %s #%d", relName(v.Tuple, v), v.Index)
}

func (s *Jump) String() string {
	// Be robust against malformed CFG.
	block := -1
	if s.block != nil && len(s.block.Succs) == 1 {
		block = s.block.Succs[0].Index
	}
	return fmt.Sprintf("jump %d", block)
}

func (s *If) String() string {
	// Be robust against malformed CFG.
	tblock, fblock := -1, -1
	if s.block != nil && len(s.block.Succs) == 2 {
		tblock = s.block.Succs[0].Index
		fblock = s.block.Succs[1].Index
	}
	return fmt.Sprintf("if %s goto %d else %d", relName(s.Cond, s), tblock, fblock)
}

func (s *Go) String() string {
	return printCall(&s.Call, "go ", s)
}

func (s *Panic) String() string {
	return "panic " + relName(s.X, s)
}

func (s *Return) String() string {
	var b bytes.Buffer
	b.WriteString("return")
	for i, r := range s.Results {
		if i == 0 {
			b.WriteString(" ")
		} else {
			b.WriteString(", ")
		}
		b.WriteString(relName(r, s))
	}
	return b.String()
}

func (*RunDefers) String() string {
	return "rundefers"
}

func (s *Send) String() string {
	return fmt.Sprintf("send %s <- %s", relName(s.Chan, s), relName(s.X, s))
}

func (s *Defer) String() string {
	return printCall(&s.Call, "defer ", s)
}

func (s *Select) String() string {
	var b bytes.Buffer
	for i, st := range s.States {
		if i > 0 {
			b.WriteString(", ")
		}
		if st.Dir == types.RecvOnly {
			b.WriteString("<-")
			b.WriteString(relName(st.Chan, s))
		} else {
			b.WriteString(relName(st.Chan, s))
			b.WriteString("<-")
			b.WriteString(relName(st.Send, s))
		}
	}
	non := ""
	if !s.Blocking {
		non = "non"
	}
	return fmt.Sprintf("select %sblocking [%s]", non, b.String())
}

func (s *Store) String() string {
	return fmt.Sprintf("*%s = %s", relName(s.Addr, s), relName(s.Val, s))
}

func (s *MapUpdate) String() string {
	return fmt.Sprintf("%s[%s] = %s", relName(s.Map, s), relName(s.Key, s), relName(s.Value, s))
}

func (s *DebugRef) String() string {
	p := s.Parent().Prog.Fset.Position(s.Pos())
	var descr interface{}
	if s.object != nil {
		descr = s.object // e.g. "var x int"
	} else {
		descr = reflect.TypeOf(s.Expr) // e.g. "*ast.CallExpr"
	}
	var addr string
	if s.IsAddr {
		addr = "address of "
	}
	return fmt.Sprintf("; %s%s @ %d:%d is %s", addr, descr, p.Line, p.Column, s.X.Name())
}

func (p *Package) String() string {
	return "package " + p.Pkg.Path()
}

var _ io.WriterTo = (*Package)(nil) // *Package implements io.Writer

func (p *Package) WriteTo(w io.Writer) (int64, error) {
	var buf bytes.Buffer
	WritePackage(&buf, p)
	n, err := w.Write(buf.Bytes())
	return int64(n), err
}

// WritePackage writes to buf a human-readable summary of p.
func WritePackage(buf *bytes.Buffer, p *Package) {
	fmt.Fprintf(buf, "%s:\n", p)

	var names []string
	maxname := 0
	for name := range p.Members {
		if l := len(name); l > maxname {
			maxname = l
		}
		names = append(names, name)
	}

	from := p.Pkg
	sort.Strings(names)
	for _, name := range names {
		switch mem := p.Members[name].(type) {
		case *NamedConst:
			fmt.Fprintf(buf, "  const %-*s %s = %s\n",
				maxname, name, mem.Name(), mem.Value.RelString(from))

		case *Function:
			fmt.Fprintf(buf, "  func  %-*s %s\n",
				maxname, name, relType(mem.Type(), from))

		case *Type:
			fmt.Fprintf(buf, "  type  %-*s %s\n",
				maxname, name, relType(mem.Type().Underlying(), from))
			for _, meth := range typeutil.IntuitiveMethodSet(mem.Type(), &p.Prog.MethodSets) {
				fmt.Fprintf(buf, "    %s\n", types.SelectionString(meth, types.RelativeTo(from)))
			}

		case *Global:
			fmt.Fprintf(buf, "  var   %-*s %s\n",
				maxname, name, relType(mem.Type().(*types.Pointer).Elem(), from))
		}
	}

	fmt.Fprintf(buf, "\n")
}

func commaOk(x bool) string {
	if x {
		return ",ok"
	}
	return ""
}

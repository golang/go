package ssa

// This file implements the String() methods for all Value and
// Instruction types.

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/types"
)

func (id Id) String() string {
	if id.Pkg == nil {
		return id.Name
	}
	return fmt.Sprintf("%s/%s", id.Pkg.Path, id.Name)
}

// relName returns the name of v relative to i.
// In most cases, this is identical to v.Name(), but for cross-package
// references to Functions (including methods) and Globals, the
// package-qualified FullName is used instead.
//
func relName(v Value, i Instruction) string {
	switch v := v.(type) {
	case *Global:
		if v.Pkg == i.Block().Func.Pkg {
			return v.Name()
		}
		return v.FullName()
	case *Function:
		if v.Pkg == nil || v.Pkg == i.Block().Func.Pkg {
			return v.Name()
		}
		return v.FullName()
	}
	return v.Name()
}

// Value.String()
//
// This method is provided only for debugging.
// It never appears in disassembly, which uses Value.Name().

func (v *Literal) String() string {
	return fmt.Sprintf("literal %s rep=%T", v.Name(), v.Value)
}

func (v *Parameter) String() string {
	return fmt.Sprintf("parameter %s : %s", v.Name(), v.Type())
}

func (v *Capture) String() string {
	return fmt.Sprintf("capture %s : %s", v.Name(), v.Type())
}

func (v *Global) String() string {
	return fmt.Sprintf("global %s : %s", v.Name(), v.Type())
}

func (v *Builtin) String() string {
	return fmt.Sprintf("builtin %s : %s", v.Name(), v.Type())
}

func (r *Function) String() string {
	return fmt.Sprintf("function %s : %s", r.Name(), r.Type())
}

// FullName returns the name of this function qualified by the
// package name, unless it is anonymous or synthetic.
//
// TODO(adonovan): move to func.go when it's submitted.
//
func (f *Function) FullName() string {
	if f.Enclosing != nil || f.Pkg == nil {
		return f.Name_ // anonymous or synthetic
	}
	return fmt.Sprintf("%s.%s", f.Pkg.ImportPath, f.Name_)
}

// FullName returns g's package-qualified name.
func (g *Global) FullName() string {
	return fmt.Sprintf("%s.%s", g.Pkg.ImportPath, g.Name_)
}

// Instruction.String()

func (v *Alloc) String() string {
	op := "local"
	if v.Heap {
		op = "new"
	}
	return fmt.Sprintf("%s %s", op, indirectType(v.Type()))
}

func (v *Phi) String() string {
	var b bytes.Buffer
	b.WriteString("phi [")
	for i, edge := range v.Edges {
		if i > 0 {
			b.WriteString(", ")
		}
		// Be robust against malformed CFG.
		blockname := "?"
		if v.Block_ != nil && i < len(v.Block_.Preds) {
			blockname = v.Block_.Preds[i].Name
		}
		b.WriteString(blockname)
		b.WriteString(": ")
		b.WriteString(relName(edge, v))
	}
	b.WriteString("]")
	return b.String()
}

func printCall(v *CallCommon, prefix string, instr Instruction) string {
	var b bytes.Buffer
	b.WriteString(prefix)
	if v.Func != nil {
		b.WriteString(relName(v.Func, instr))
	} else {
		name := underlyingType(v.Recv.Type()).(*types.Interface).Methods[v.Method].Name
		fmt.Fprintf(&b, "invoke %s.%s [#%d]", relName(v.Recv, instr), name, v.Method)
	}
	b.WriteString("(")
	for i, arg := range v.Args {
		if i > 0 {
			b.WriteString(", ")
		}
		b.WriteString(relName(arg, instr))
	}
	if v.HasEllipsis {
		b.WriteString("...")
	}
	b.WriteString(")")
	return b.String()
}

func (v *Call) String() string {
	return printCall(&v.CallCommon, "", v)
}

func (v *BinOp) String() string {
	return fmt.Sprintf("%s %s %s", relName(v.X, v), v.Op.String(), relName(v.Y, v))
}

func (v *UnOp) String() string {
	return fmt.Sprintf("%s%s%s", v.Op, relName(v.X, v), commaOk(v.CommaOk))
}

func (v *Conv) String() string {
	return fmt.Sprintf("convert %s <- %s (%s)", v.Type(), v.X.Type(), relName(v.X, v))
}

func (v *ChangeInterface) String() string {
	return fmt.Sprintf("change interface %s <- %s (%s)", v.Type(), v.X.Type(), relName(v.X, v))
}

func (v *MakeInterface) String() string {
	return fmt.Sprintf("make interface %s <- %s (%s)", v.Type(), v.X.Type(), relName(v.X, v))
}

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
	var b bytes.Buffer
	b.WriteString("make slice ")
	b.WriteString(v.Type().String())
	b.WriteString(" ")
	b.WriteString(relName(v.Len, v))
	b.WriteString(" ")
	b.WriteString(relName(v.Cap, v))
	return b.String()
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
	b.WriteString("]")
	return b.String()
}

func (v *MakeMap) String() string {
	res := ""
	if v.Reserve != nil {
		res = relName(v.Reserve, v)
	}
	return fmt.Sprintf("make %s %s", v.Type(), res)
}

func (v *MakeChan) String() string {
	return fmt.Sprintf("make %s %s", v.Type(), relName(v.Size, v))
}

func (v *FieldAddr) String() string {
	fields := underlyingType(indirectType(v.X.Type())).(*types.Struct).Fields
	// Be robust against a bad index.
	name := "?"
	if v.Field >= 0 && v.Field < len(fields) {
		name = fields[v.Field].Name
	}
	return fmt.Sprintf("&%s.%s [#%d]", relName(v.X, v), name, v.Field)
}

func (v *Field) String() string {
	fields := underlyingType(v.X.Type()).(*types.Struct).Fields
	// Be robust against a bad index.
	name := "?"
	if v.Field >= 0 && v.Field < len(fields) {
		name = fields[v.Field].Name
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
	return fmt.Sprintf("typeassert%s %s.(%s)", commaOk(v.CommaOk), relName(v.X, v), v.AssertedType)
}

func (v *Extract) String() string {
	return fmt.Sprintf("extract %s #%d", relName(v.Tuple, v), v.Index)
}

func (s *Jump) String() string {
	// Be robust against malformed CFG.
	blockname := "?"
	if s.Block_ != nil && len(s.Block_.Succs) == 1 {
		blockname = s.Block_.Succs[0].Name
	}
	return fmt.Sprintf("jump %s", blockname)
}

func (s *If) String() string {
	// Be robust against malformed CFG.
	tblockname, fblockname := "?", "?"
	if s.Block_ != nil && len(s.Block_.Succs) == 2 {
		tblockname = s.Block_.Succs[0].Name
		fblockname = s.Block_.Succs[1].Name
	}
	return fmt.Sprintf("if %s goto %s else %s", relName(s.Cond, s), tblockname, fblockname)
}

func (s *Go) String() string {
	return printCall(&s.CallCommon, "go ", s)
}

func (s *Ret) String() string {
	var b bytes.Buffer
	b.WriteString("ret")
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

func (s *Send) String() string {
	return fmt.Sprintf("send %s <- %s", relName(s.Chan, s), relName(s.X, s))
}

func (s *Defer) String() string {
	return printCall(&s.CallCommon, "defer ", s)
}

func (s *Select) String() string {
	var b bytes.Buffer
	for i, st := range s.States {
		if i > 0 {
			b.WriteString(", ")
		}
		if st.Dir == ast.RECV {
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

func (p *Package) String() string {
	// TODO(adonovan): prettify output.
	var b bytes.Buffer
	fmt.Fprintf(&b, "Package %s at %s:\n", p.ImportPath, p.Prog.Files.File(p.Pos).Name())

	// TODO(adonovan): make order deterministic.
	maxname := 0
	for name := range p.Members {
		if l := len(name); l > maxname {
			maxname = l
		}
	}

	for name, mem := range p.Members {
		switch mem := mem.(type) {
		case *Literal:
			fmt.Fprintf(&b, " const %-*s %s\n", maxname, name, mem.Name())

		case *Function:
			fmt.Fprintf(&b, " func  %-*s %s\n", maxname, name, mem.Type())

		case *Type:
			fmt.Fprintf(&b, " type  %-*s %s\n", maxname, name, mem.NamedType.Underlying)
			// TODO(adonovan): make order deterministic.
			for name, method := range mem.Methods {
				fmt.Fprintf(&b, "       method %s %s\n", name, method.Signature)
			}

		case *Global:
			fmt.Fprintf(&b, " var   %-*s %s\n", maxname, name, mem.Type())

		}
	}
	return b.String()
}

func commaOk(x bool) string {
	if x {
		return ",ok"
	}
	return ""
}

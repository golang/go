// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package asmgen

import (
	"fmt"
	"slices"
	"strings"
)

// Note: Exported fields and methods are expected to be used
// by function generators (like the ones in add.go and so on).
// Unexported fields and methods should not be.

// A Func represents a single assembly function.
type Func struct {
	Name    string
	Asm     *Asm
	inputs  []string       // name of input slices (not beginning with z)
	outputs []string       // names of output slices (beginning with z)
	args    map[string]int // offsets of args, results on stack
}

// Func starts a new function in the assembly output.
func (a *Asm) Func(decl string) *Func {
	d, ok := strings.CutPrefix(decl, "func ")
	if !ok {
		a.Fatalf("func decl does not begin with 'func '")
	}
	name, d, ok := strings.Cut(d, "(")
	if !ok {
		a.Fatalf("func decl does not have func arg list")
	}
	f := &Func{
		Name: name,
		Asm:  a,
		args: make(map[string]int),
	}
	a.FreeAll()

	// Parse argument names and types. Quick and dirty.
	// Convert (args) (results) into args, results.
	d = strings.ReplaceAll(d, ") (", ", ")
	d = strings.TrimSuffix(d, ")")
	args := strings.Split(d, ",")

	// Assign implicit types to all arguments (x, y int -> x int, y int).
	typ := ""
	for i, arg := range slices.Backward(args) {
		arg = strings.TrimSpace(arg)
		if !strings.Contains(arg, " ") {
			if typ == "" {
				a.Fatalf("missing argument type")
			}
			arg += " " + typ
		} else {
			_, typ, _ = strings.Cut(arg, " ")
		}
		args[i] = arg
	}

	// Record mapping from names to offsets.
	off := 0
	for _, arg := range args {
		name, typ, _ := strings.Cut(arg, " ")
		switch typ {
		default:
			a.Fatalf("unknown type %s", typ)
		case "Word", "uint", "int":
			f.args[name] = off
			off += a.Arch.WordBytes
		case "[]Word":
			if strings.HasPrefix(name, "z") {
				f.outputs = append(f.outputs, name)
			} else {
				f.inputs = append(f.inputs, name)
			}
			f.args[name+"_base"] = off
			f.args[name+"_len"] = off + a.Arch.WordBytes
			f.args[name+"_cap"] = off + 2*a.Arch.WordBytes
			off += 3 * a.Arch.WordBytes
		}
	}

	a.Printf("\n")
	a.Printf("// %s\n", decl)
	a.Printf("TEXT ·%s(SB), NOSPLIT, $0\n", name)
	if a.Arch.setup != nil {
		a.Arch.setup(f)
	}
	return f
}

// Arg allocates a new register, copies the named argument (or result) into it,
// and returns that register.
func (f *Func) Arg(name string) Reg {
	return f.ArgHint(name, HintNone)
}

// ArgHint is like Arg but uses a register allocation hint.
func (f *Func) ArgHint(name string, hint Hint) Reg {
	off, ok := f.args[name]
	if !ok {
		f.Asm.Fatalf("unknown argument %s", name)
	}
	mem := Reg{fmt.Sprintf("%s+%d(FP)", name, off)}
	if hint == HintMemOK && f.Asm.Arch.memOK {
		return mem
	}
	r := f.Asm.RegHint(hint)
	f.Asm.Mov(mem, r)
	return r
}

// ArgPtr is like Arg but returns a RegPtr.
func (f *Func) ArgPtr(name string) RegPtr {
	return RegPtr(f.Arg(name))
}

// StoreArg stores src into the named argument (or result).
func (f *Func) StoreArg(src Reg, name string) {
	off, ok := f.args[name]
	if !ok {
		f.Asm.Fatalf("unknown argument %s", name)
	}
	a := f.Asm
	mem := Reg{fmt.Sprintf("%s+%d(FP)", name, off)}
	if src.IsImm() && !a.Arch.memOK {
		r := a.Reg()
		a.Mov(src, r)
		a.Mov(r, mem)
		a.Free(r)
		return
	}
	a.Mov(src, mem)
}

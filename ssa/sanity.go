// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// An optional pass for sanity-checking invariants of the SSA representation.
// Currently it checks CFG invariants but little at the instruction level.

import (
	"fmt"
	"io"
	"os"
	"strings"

	"code.google.com/p/go.tools/go/types"
)

type sanity struct {
	reporter io.Writer
	fn       *Function
	block    *BasicBlock
	insane   bool
}

// sanityCheck performs integrity checking of the SSA representation
// of the function fn and returns true if it was valid.  Diagnostics
// are written to reporter if non-nil, os.Stderr otherwise.  Some
// diagnostics are only warnings and do not imply a negative result.
//
// Sanity-checking is intended to facilitate the debugging of code
// transformation passes.
//
func sanityCheck(fn *Function, reporter io.Writer) bool {
	if reporter == nil {
		reporter = os.Stderr
	}
	return (&sanity{reporter: reporter}).checkFunction(fn)
}

// mustSanityCheck is like sanityCheck but panics instead of returning
// a negative result.
//
func mustSanityCheck(fn *Function, reporter io.Writer) {
	if !sanityCheck(fn, reporter) {
		fn.DumpTo(os.Stderr)
		panic("SanityCheck failed")
	}
}

func (s *sanity) diagnostic(prefix, format string, args ...interface{}) {
	fmt.Fprintf(s.reporter, "%s: function %s", prefix, s.fn)
	if s.block != nil {
		fmt.Fprintf(s.reporter, ", block %s", s.block)
	}
	io.WriteString(s.reporter, ": ")
	fmt.Fprintf(s.reporter, format, args...)
	io.WriteString(s.reporter, "\n")
}

func (s *sanity) errorf(format string, args ...interface{}) {
	s.insane = true
	s.diagnostic("Error", format, args...)
}

func (s *sanity) warnf(format string, args ...interface{}) {
	s.diagnostic("Warning", format, args...)
}

// findDuplicate returns an arbitrary basic block that appeared more
// than once in blocks, or nil if all were unique.
func findDuplicate(blocks []*BasicBlock) *BasicBlock {
	if len(blocks) < 2 {
		return nil
	}
	if blocks[0] == blocks[1] {
		return blocks[0]
	}
	// Slow path:
	m := make(map[*BasicBlock]bool)
	for _, b := range blocks {
		if m[b] {
			return b
		}
		m[b] = true
	}
	return nil
}

func (s *sanity) checkInstr(idx int, instr Instruction) {
	switch instr := instr.(type) {
	case *If, *Jump, *Ret, *Panic:
		s.errorf("control flow instruction not at end of block")
	case *Phi:
		if idx == 0 {
			// It suffices to apply this check to just the first phi node.
			if dup := findDuplicate(s.block.Preds); dup != nil {
				s.errorf("phi node in block with duplicate predecessor %s", dup)
			}
		} else {
			prev := s.block.Instrs[idx-1]
			if _, ok := prev.(*Phi); !ok {
				s.errorf("Phi instruction follows a non-Phi: %T", prev)
			}
		}
		if ne, np := len(instr.Edges), len(s.block.Preds); ne != np {
			s.errorf("phi node has %d edges but %d predecessors", ne, np)

		} else {
			for i, e := range instr.Edges {
				if e == nil {
					s.errorf("phi node '%s' has no value for edge #%d from %s", instr.Comment, i, s.block.Preds[i])
				}
			}
		}

	case *Alloc:
		if !instr.Heap {
			found := false
			for _, l := range s.fn.Locals {
				if l == instr {
					found = true
					break
				}
			}
			if !found {
				s.errorf("local alloc %s = %s does not appear in Function.Locals", instr.Name(), instr)
			}
		}

	case *BinOp:
	case *Call:
	case *ChangeInterface:
	case *ChangeType:
	case *Convert:
		if _, ok := instr.X.Type().Underlying().(*types.Basic); !ok {
			if _, ok := instr.Type().Underlying().(*types.Basic); !ok {
				s.errorf("convert %s -> %s: at least one type must be basic", instr.X.Type(), instr.Type())
			}
		}
	case *Defer:
	case *Extract:
	case *Field:
	case *FieldAddr:
	case *Go:
	case *Index:
	case *IndexAddr:
	case *Lookup:
	case *MakeChan:
	case *MakeClosure:
		numFree := len(instr.Fn.(*Function).FreeVars)
		numBind := len(instr.Bindings)
		if numFree != numBind {
			s.errorf("MakeClosure has %d Bindings for function %s with %d free vars",
				numBind, instr.Fn, numFree)

		}
		if recv := instr.Type().(*types.Signature).Recv(); recv != nil {
			s.errorf("MakeClosure's type includes receiver %s", recv.Type())
		}

	case *MakeInterface:
	case *MakeMap:
	case *MakeSlice:
	case *MapUpdate:
	case *Next:
	case *Range:
	case *RunDefers:
	case *Select:
	case *Send:
	case *Slice:
	case *Store:
	case *TypeAssert:
	case *UnOp:
	case *DebugRef:
		// TODO(adonovan): implement checks.
	default:
		panic(fmt.Sprintf("Unknown instruction type: %T", instr))
	}

	// Check that value-defining instructions have valid types.
	if v, ok := instr.(Value); ok {
		t := v.Type()
		if t == nil {
			s.errorf("no type: %s = %s", v.Name(), v)
		} else if t == tRangeIter {
			// not a proper type; ignore.
		} else if b, ok := t.Underlying().(*types.Basic); ok && b.Info()&types.IsUntyped != 0 {
			s.errorf("instruction has 'untyped' result: %s = %s : %s", v.Name(), v, t)
		}
	}

	// TODO(adonovan): sanity-check Consts used as instruction Operands(),
	// e.g. reject Consts with "untyped" types.
	//
	// All other non-Instruction Values can be found via their
	// enclosing Function or Package.
}

func (s *sanity) checkFinalInstr(idx int, instr Instruction) {
	switch instr.(type) {
	case *If:
		if nsuccs := len(s.block.Succs); nsuccs != 2 {
			s.errorf("If-terminated block has %d successors; expected 2", nsuccs)
			return
		}
		if s.block.Succs[0] == s.block.Succs[1] {
			s.errorf("If-instruction has same True, False target blocks: %s", s.block.Succs[0])
			return
		}

	case *Jump:
		if nsuccs := len(s.block.Succs); nsuccs != 1 {
			s.errorf("Jump-terminated block has %d successors; expected 1", nsuccs)
			return
		}

	case *Ret:
		if nsuccs := len(s.block.Succs); nsuccs != 0 {
			s.errorf("Ret-terminated block has %d successors; expected none", nsuccs)
			return
		}
		// TODO(adonovan): check number and types of results

	case *Panic:
		if nsuccs := len(s.block.Succs); nsuccs != 0 {
			s.errorf("Panic-terminated block has %d successors; expected none", nsuccs)
			return
		}

	default:
		s.errorf("non-control flow instruction at end of block")
	}
}

func (s *sanity) checkBlock(b *BasicBlock, index int) {
	s.block = b

	if b.Index != index {
		s.errorf("block has incorrect Index %d", b.Index)
	}
	if b.parent != s.fn {
		s.errorf("block has incorrect parent %s", b.parent)
	}

	// Check all blocks are reachable.
	// (The entry block is always implicitly reachable.)
	if index > 0 && len(b.Preds) == 0 {
		s.warnf("unreachable block")
		if b.Instrs == nil {
			// Since this block is about to be pruned,
			// tolerating transient problems in it
			// simplifies other optimizations.
			return
		}
	}

	// Check predecessor and successor relations are dual,
	// and that all blocks in CFG belong to same function.
	for _, a := range b.Preds {
		found := false
		for _, bb := range a.Succs {
			if bb == b {
				found = true
				break
			}
		}
		if !found {
			s.errorf("expected successor edge in predecessor %s; found only: %s", a, a.Succs)
		}
		if a.parent != s.fn {
			s.errorf("predecessor %s belongs to different function %s", a, a.parent)
		}
	}
	for _, c := range b.Succs {
		found := false
		for _, bb := range c.Preds {
			if bb == b {
				found = true
				break
			}
		}
		if !found {
			s.errorf("expected predecessor edge in successor %s; found only: %s", c, c.Preds)
		}
		if c.parent != s.fn {
			s.errorf("successor %s belongs to different function %s", c, c.parent)
		}
	}

	// Check each instruction is sane.
	// TODO(adonovan): check Instruction invariants:
	// - check Operands is dual to Value.Referrers.
	// - check all Operands that are also Instructions belong to s.fn too
	//   (and for bonus marks, that their block dominates block b).
	n := len(b.Instrs)
	if n == 0 {
		s.errorf("basic block contains no instructions")
	}
	for j, instr := range b.Instrs {
		if instr == nil {
			s.errorf("nil instruction at index %d", j)
			continue
		}
		if b2 := instr.Block(); b2 == nil {
			s.errorf("nil Block() for instruction at index %d", j)
			continue
		} else if b2 != b {
			s.errorf("wrong Block() (%s) for instruction at index %d ", b2, j)
			continue
		}
		if j < n-1 {
			s.checkInstr(j, instr)
		} else {
			s.checkFinalInstr(j, instr)
		}
	}
}

func (s *sanity) checkFunction(fn *Function) bool {
	// TODO(adonovan): check Function invariants:
	// - check params match signature
	// - check transient fields are nil
	// - warn if any fn.Locals do not appear among block instructions.
	s.fn = fn
	if fn.Prog == nil {
		s.errorf("nil Prog")
	}
	// All functions have a package, except wrappers for error.Error()
	// (and embedding of that method in other interfaces).
	if fn.Pkg == nil {
		if strings.Contains(fn.Synthetic, "wrapper") &&
			strings.HasSuffix(fn.name, "Error") {
			// wrapper for error.Error() has no package.
		} else {
			s.errorf("nil Pkg")
		}
	}
	for i, l := range fn.Locals {
		if l.Parent() != fn {
			s.errorf("Local %s at index %d has wrong parent", l.Name(), i)
		}
		if l.Heap {
			s.errorf("Local %s at index %d has Heap flag set", l.Name(), i)
		}
	}
	for i, p := range fn.Params {
		if p.Parent() != fn {
			s.errorf("Param %s at index %d has wrong parent", p.Name(), i)
		}
	}
	for i, fv := range fn.FreeVars {
		if fv.Parent() != fn {
			s.errorf("FreeVar %s at index %d has wrong parent", fv.Name(), i)
		}
	}

	if fn.Blocks != nil && len(fn.Blocks) == 0 {
		// Function _had_ blocks (so it's not external) but
		// they were "optimized" away, even the entry block.
		s.errorf("Blocks slice is non-nil but empty")
	}
	for i, b := range fn.Blocks {
		if b == nil {
			s.warnf("nil *BasicBlock at f.Blocks[%d]", i)
			continue
		}
		s.checkBlock(b, i)
	}
	s.block = nil
	s.fn = nil
	return !s.insane
}

// sanityCheckPackage checks invariants of packages upon creation.
// It does not require that the package is built.
// Unlike sanityCheck (for functions), it just panics at the first error.
func sanityCheckPackage(pkg *Package) {
	for name, mem := range pkg.Members {
		if name != mem.Name() {
			panic(fmt.Sprintf("%s: %T.Name() = %s, want %s",
				pkg.Object.Path(), mem, mem.Name(), name))
		}
		obj := mem.Object()
		if obj == nil {
			// This check is sound because fields
			// {Global,Function}.object have type
			// types.Object.  (If they were declared as
			// *types.{Var,Func}, we'd have a non-empty
			// interface containing a nil pointer.)

			continue // not all members have typechecker objects
		}
		if obj.Name() != name {
			panic(fmt.Sprintf("%s: %T.Object().Name() = %s, want %s",
				pkg.Object.Path(), mem, obj.Name(), name))
		}
		if obj.Pos() != mem.Pos() {
			panic(fmt.Sprintf("%s Pos=%d obj.Pos=%d", mem, mem.Pos(), obj.Pos()))
		}
	}
}

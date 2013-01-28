package ssa

// An optional pass for sanity checking invariants of the SSA representation.
// Currently it checks CFG invariants but little at the instruction level.

import (
	"bytes"
	"fmt"
	"io"
	"os"
)

type sanity struct {
	reporter io.Writer
	fn       *Function
	block    *BasicBlock
	insane   bool
}

// SanityCheck performs integrity checking of the SSA representation
// of the function fn and returns true if it was valid.  Diagnostics
// are written to reporter if non-nil, os.Stderr otherwise.  Some
// diagnostics are only warnings and do not imply a negative result.
//
// Sanity checking is intended to facilitate the debugging of code
// transformation passes.
//
func SanityCheck(fn *Function, reporter io.Writer) bool {
	if reporter == nil {
		reporter = os.Stderr
	}
	return (&sanity{reporter: reporter}).checkFunction(fn)
}

// MustSanityCheck is like SanityCheck but panics instead of returning
// a negative result.
//
func MustSanityCheck(fn *Function, reporter io.Writer) {
	if !SanityCheck(fn, reporter) {
		panic("SanityCheck failed")
	}
}

// blockNames returns the names of the specified blocks as a
// human-readable string.
//
func blockNames(blocks []*BasicBlock) string {
	var buf bytes.Buffer
	for i, b := range blocks {
		if i > 0 {
			io.WriteString(&buf, ", ")
		}
		io.WriteString(&buf, b.Name)
	}
	return buf.String()
}

func (s *sanity) diagnostic(prefix, format string, args ...interface{}) {
	fmt.Fprintf(s.reporter, "%s: function %s", prefix, s.fn.FullName())
	if s.block != nil {
		fmt.Fprintf(s.reporter, ", block %s", s.block.Name)
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
	case *If, *Jump, *Ret:
		s.errorf("control flow instruction not at end of block")
	case *Phi:
		if idx == 0 {
			// It suffices to apply this check to just the first phi node.
			if dup := findDuplicate(s.block.Preds); dup != nil {
				s.errorf("phi node in block with duplicate predecessor %s", dup.Name)
			}
		} else {
			prev := s.block.Instrs[idx-1]
			if _, ok := prev.(*Phi); !ok {
				s.errorf("Phi instruction follows a non-Phi: %T", prev)
			}
		}
		if ne, np := len(instr.Edges), len(s.block.Preds); ne != np {
			s.errorf("phi node has %d edges but %d predecessors", ne, np)
		}

	case *Alloc:
	case *Call:
	case *BinOp:
	case *UnOp:
	case *MakeClosure:
	case *MakeChan:
	case *MakeMap:
	case *MakeSlice:
	case *Slice:
	case *Field:
	case *FieldAddr:
	case *IndexAddr:
	case *Index:
	case *Select:
	case *Range:
	case *TypeAssert:
	case *Extract:
	case *Go:
	case *Defer:
	case *Send:
	case *Store:
	case *MapUpdate:
	case *Next:
	case *Lookup:
	case *Conv:
	case *ChangeInterface:
	case *MakeInterface:
		// TODO(adonovan): implement checks.
	default:
		panic(fmt.Sprintf("Unknown instruction type: %T", instr))
	}
}

func (s *sanity) checkFinalInstr(idx int, instr Instruction) {
	switch instr.(type) {
	case *If:
		if nsuccs := len(s.block.Succs); nsuccs != 2 {
			s.errorf("If-terminated block has %d successors; expected 2", nsuccs)
			return
		}
		if s.block.Succs[0] == s.block.Succs[1] {
			s.errorf("If-instruction has same True, False target blocks: %s", s.block.Succs[0].Name)
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

	default:
		s.errorf("non-control flow instruction at end of block")
	}
}

func (s *sanity) checkBlock(b *BasicBlock, isEntry bool) {
	s.block = b

	// Check all blocks are reachable.
	// (The entry block is always implicitly reachable.)
	if !isEntry && len(b.Preds) == 0 {
		s.warnf("unreachable block")
		if b.Instrs == nil {
			// Since this block is about to be pruned,
			// tolerating transient problems in it
			// simplifies other optimisations.
			return
		}
	}

	// Check predecessor and successor relations are dual.
	for _, a := range b.Preds {
		found := false
		for _, bb := range a.Succs {
			if bb == b {
				found = true
				break
			}
		}
		if !found {
			s.errorf("expected successor edge in predecessor %s; found only: %s", a.Name, blockNames(a.Succs))
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
			s.errorf("expected predecessor edge in successor %s; found only: %s", c.Name, blockNames(c.Preds))
		}
	}

	// Check each instruction is sane.
	n := len(b.Instrs)
	if n == 0 {
		s.errorf("basic block contains no instructions")
	}
	for j, instr := range b.Instrs {
		if b2 := instr.Block(); b2 == nil {
			s.errorf("nil Block() for instruction at index %d", j)
			continue
		} else if b2 != b {
			s.errorf("wrong Block() (%s) for instruction at index %d ", b2.Name, j)
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
	// - check owning Package (if any) contains this function.
	// - check params match signature
	// - check locals are all !Heap
	// - check transient fields are nil
	// - check block labels are unique (warning)
	s.fn = fn
	if fn.Prog == nil {
		s.errorf("nil Prog")
	}
	for i, b := range fn.Blocks {
		if b == nil {
			s.warnf("nil *BasicBlock at f.Blocks[%d]", i)
			continue
		}
		s.checkBlock(b, i == 0)
	}
	s.block = nil
	s.fn = nil
	return !s.insane
}

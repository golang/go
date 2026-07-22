// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/ssa/block"
	"cmd/compile/internal/ssa/ssacore"
	"cmd/internal/src"
)

// deadcode removes dead code from f.
func deadcode(f *ssacore.Func) {
	// deadcode after regalloc is forbidden for now. Regalloc
	// doesn't quite generate legal SSA which will lead to some
	// required moves being eliminated. See the comment at the
	// top of regalloc.go for details.
	if f.RegAlloc != nil {
		f.Fatalf("deadcode after regalloc")
	}

	// Find reachable blocks.
	reachable := ssacore.ReachableBlocks(f)

	// Get rid of edges from dead to live code.
	for _, b := range f.Blocks {
		if reachable[b.ID] {
			continue
		}
		for i := 0; i < len(b.Succs); {
			e := b.Succs[i]
			if reachable[e.B.ID] {
				b.RemoveEdge(i)
			} else {
				i++
			}
		}
	}

	// Get rid of dead edges from live code.
	for _, b := range f.Blocks {
		if !reachable[b.ID] {
			continue
		}
		if b.Kind != block.BlockFirst {
			continue
		}
		b.RemoveEdge(1)
		b.Kind = block.BlockPlain
		b.Likely = ssacore.BranchUnknown
	}

	// Splice out any copies introduced during dead block removal.
	copyelim(f)

	// Find live values.
	live, order := ssacore.LiveValues(f, reachable)
	defer func() { f.Cache.FreeBoolSlice(live) }()
	defer func() { f.Cache.FreeValueSlice(order) }()

	// Remove dead & duplicate entries from namedValues map.
	s := f.NewSparseSet(f.NumValues())
	defer f.RetSparseSet(s)
	i := 0
	for _, name := range f.Names {
		j := 0
		s.Clear()
		values := f.NamedValues[name]
		for _, v := range values {
			if live[v.ID] && !s.Contains(v.ID) {
				values[j] = v
				j++
				s.Add(v.ID)
			}
		}
		if j == 0 {
			delete(f.NamedValues, name)
		} else {
			f.Names[i] = name
			i++
			for k := len(values) - 1; k >= j; k-- {
				values[k] = nil
			}
			f.NamedValues[name] = values[:j]
		}
	}
	clear(f.Names[i:])
	f.Names = f.Names[:i]

	pendingLines := f.CachedLineStarts // Holds statement boundaries that need to be moved to a new value/block
	pendingLines.Clear()

	// Unlink values and conserve statement boundaries
	for i, b := range f.Blocks {
		if !reachable[b.ID] {
			// TODO what if control is statement boundary? Too late here.
			b.ResetControls()
		}
		for _, v := range b.Values {
			if !live[v.ID] {
				v.ResetArgs()
				if v.Pos.IsStmt() == src.PosIsStmt && reachable[b.ID] {
					pendingLines.Set(v.Pos, int32(i)) // TODO could be more than one pos for a line
				}
			}
		}
	}

	// Find new homes for lost lines -- require earliest in data flow with same line that is also in same block
	for i := len(order) - 1; i >= 0; i-- {
		w := order[i]
		if j, ok := pendingLines.Get(w.Pos); ok && f.Blocks[j] == w.Block {
			w.Pos = w.Pos.WithIsStmt()
			pendingLines.Remove(w.Pos)
		}
	}

	// Any boundary that failed to match a live value can move to a block end
	pendingLines.ForeachEntry(func(j int32, l uint, bi int32) {
		b := f.Blocks[bi]
		if b.Pos.Line() == l && b.Pos.FileIndex() == j {
			b.Pos = b.Pos.WithIsStmt()
		}
	})

	// Remove dead values from blocks' value list. Return dead
	// values to the allocator.
	for _, b := range f.Blocks {
		i := 0
		for _, v := range b.Values {
			if live[v.ID] {
				b.Values[i] = v
				i++
			} else {
				f.FreeValue(v)
			}
		}
		b.TruncateValues(i)
	}

	// Remove unreachable blocks. Return dead blocks to allocator.
	i = 0
	for _, b := range f.Blocks {
		if reachable[b.ID] {
			f.Blocks[i] = b
			i++
		} else {
			if len(b.Values) > 0 {
				b.Fatalf("live values in unreachable block %v: %v", b, b.Values)
			}
			f.FreeBlock(b)
		}
	}
	// zero remainder to help GC
	clear(f.Blocks[i:])
	f.Blocks = f.Blocks[:i]
}

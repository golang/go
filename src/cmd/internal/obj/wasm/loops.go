// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package wasm

import (
	"cmd/internal/obj"
	"internal/abi"
)

// structureLoops turns loops that contain no calls into WebAssembly loops.
// A jump to an earlier basic block has to go through entryPointLoop, since a
// goroutine can resume at any resume point. A loop without calls is never
// resumed in, so its basic blocks can branch to each other directly.
func structureLoops(s *obj.LSym, newprog obj.ProgAlloc) {
	if s.Func().FuncFlag&abi.FuncFlagAsm != 0 {
		return
	}

	type block struct {
		first, last    *obj.Prog // last is the resume point
		jumps          []*obj.Prog
		call           bool
		minSrc, maxSrc int       // the basic blocks that jump here are in this range
		loop           *obj.Prog // Loop starting here
		scope          *obj.Prog // Block ending here
	}
	var blocks []block
	index := make(map[*obj.Prog]int) // first instruction -> basic block
	start := true
	for p := s.Func().Text; p != nil; p = p.Link {
		if start {
			index[p] = len(blocks)
			blocks = append(blocks, block{first: p, minSrc: len(blocks), maxSrc: len(blocks) - 1})
			start = false
		}
		b := &blocks[len(blocks)-1]
		switch p.As {
		case obj.ACALL:
			b.call = true
		case obj.AJMP:
			if p.To.Type == obj.TYPE_BRANCH {
				b.jumps = append(b.jumps, p)
			}
		case ARESUMEPOINT:
			b.last = p
			start = true
		}
	}

	target := func(p *obj.Prog) int {
		v, ok := index[p.To.Val.(*obj.Prog)]
		if !ok {
			panic("jump into the middle of a basic block")
		}
		return v
	}
	for u := range blocks {
		for _, p := range blocks[u].jumps {
			b := &blocks[target(p)]
			b.minSrc = min(b.minSrc, u)
			b.maxSrc = max(b.maxSrc, u)
		}
	}

	insert := func(p *obj.Prog, as obj.As) *obj.Prog {
		q := obj.Appendp(p, newprog)
		q.As = as
		return q
	}

	// isLoop reports whether basic blocks h to e can become a WebAssembly loop.
	isLoop := func(h, e int) bool {
		for i := h; i <= e; i++ {
			b := &blocks[i]
			if b.call {
				return false
			}
			if i == h {
				continue
			}
			if b.minSrc < h || b.maxSrc > e {
				return false // entered from outside the loop
			}
			if b.maxSrc >= i && b.loop == nil {
				return false // backward jump that is not a WebAssembly loop
			}
		}
		return true
	}

	// Go backwards, so that the loops inside a loop come before it.
	found := false
	for h := len(blocks) - 1; h > 0; h-- {
		e := blocks[h].maxSrc // last basic block that jumps back to h
		if e < h || !isLoop(h, e) {
			continue
		}
		found = true

		// A basic block that is jumped to from an earlier one in the loop needs
		// a Block ending at it, unless a loop inside has added it already.
		// The Blocks start at the Loop, the one ending last first.
		loop := insert(blocks[h-1].last, ALoop)
		loop.Pos = blocks[h].first.Pos
		blocks[h].loop = loop
		p := loop
		for i := e; i > h; i-- {
			end := blocks[i-1].last
			if end.As != ARESUMEPOINT || blocks[i].minSrc == i {
				continue
			}
			p = insert(p, ABlock)
			blocks[i].scope = p
			end.As = AEnd
		}
		// The resume point after the last basic block stays.
		blocks[e].last.As = AEnd
		blocks[e].last = insert(blocks[e].last, ARESUMEPOINT)

		for u := h; u <= e; u++ {
			for _, p := range blocks[u].jumps {
				if p.As != obj.AJMP {
					continue
				}
				v := target(p)
				switch {
				case v == h:
					p.As = ABr
					p.To.Val = loop
				case v > h && v <= e:
					p.As = ABr
					p.To.Val = blocks[v].scope
				}
			}
		}
	}
	if !found {
		return
	}

	// The first instruction of a loop may be a resume point that is gone.
	for u := range blocks {
		for _, p := range blocks[u].jumps {
			if p.As != obj.AJMP {
				continue
			}
			if loop := blocks[target(p)].loop; loop != nil {
				p.To.Val = loop
			}
		}
	}

	// Remove the resume points left inside the loops.
	depth := 0
	for p := s.Func().Text; p.Link != nil; {
		q := p.Link
		switch q.As {
		case ABlock, ALoop, AIf:
			depth++
		case AEnd:
			depth--
		case ARESUMEPOINT:
			if depth > 0 {
				p.Link = q.Link
				continue
			}
		}
		p = q
	}
}

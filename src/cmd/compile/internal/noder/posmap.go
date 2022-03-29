// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/syntax"
	"cmd/internal/src"
)

// A posMap handles mapping from syntax.Pos to src.XPos.
type posMap struct {
	bases map[*syntax.PosBase]*src.PosBase
	cache struct {
		last *syntax.PosBase
		base *src.PosBase
	}
}

type poser interface{ Pos() syntax.Pos }
type ender interface{ End() syntax.Pos }

func (m *posMap) pos(p poser) src.XPos { return m.makeXPos(p.Pos()) }
func (m *posMap) end(p ender) src.XPos { return m.makeXPos(p.End()) }

func (m *posMap) makeXPos(pos syntax.Pos) src.XPos {
	if !pos.IsKnown() {
		// TODO(mdempsky): Investigate restoring base.Fatalf.
		return src.NoXPos
	}

	posBase := m.makeSrcPosBase(pos.Base())
	return base.Ctxt.PosTable.XPos(src.MakePos(posBase, pos.Line(), pos.Col()))
}

// makeSrcPosBase translates from a *syntax.PosBase to a *src.PosBase.
func (m *posMap) makeSrcPosBase(b0 *syntax.PosBase) *src.PosBase {
	// fast path: most likely PosBase hasn't changed
	if m.cache.last == b0 {
		return m.cache.base
	}

	b1, ok := m.bases[b0]
	if !ok {
		fn := b0.Filename()
		absfn := trimFilename(b0)

		if b0.IsFileBase() {
			b1 = src.NewFileBase(fn, absfn)
		} else {
			// line directive base
			p0 := b0.Pos()
			p0b := p0.Base()
			if p0b == b0 {
				panic("infinite recursion in makeSrcPosBase")
			}
			p1 := src.MakePos(m.makeSrcPosBase(p0b), p0.Line(), p0.Col())
			b1 = src.NewLinePragmaBase(p1, fn, absfn, b0.Line(), b0.Col())
		}
		if m.bases == nil {
			m.bases = make(map[*syntax.PosBase]*src.PosBase)
		}
		m.bases[b0] = b1
	}

	// update cache
	m.cache.last = b0
	m.cache.base = b1

	return b1
}

func (m *posMap) join(other *posMap) {
	if m.bases == nil {
		m.bases = make(map[*syntax.PosBase]*src.PosBase)
	}
	for k, v := range other.bases {
		if m.bases[k] != nil {
			base.Fatalf("duplicate posmap bases")
		}
		m.bases[k] = v
	}
}

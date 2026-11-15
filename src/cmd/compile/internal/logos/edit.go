// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"cmp"
	"fmt"
	"slices"
	"sort"
	"strings"

	"cmd/compile/internal/syntax"
	"cmd/internal/logos/editmap"
)

type Edit struct {
	Start, End syntax.Pos
	Text       string
	Chunks     []chunk
}

type splicePlacement uint8

const (
	placementOrdinary splicePlacement = iota
	placementAfterExpression
)

type splice struct {
	start, end int
	text       string
	chunks     []chunk
	owner      int
	placement  splicePlacement
	host       *cfHost
}

func lineStarts(src []byte) []int {
	starts := []int{0}
	for i, c := range src {
		if c == '\n' {
			starts = append(starts, i+1)
		}
	}
	return starts
}

func offsetOf(starts []int, size int, pos syntax.Pos) (int, error) {
	return lineColOffset(starts, size, int(pos.Line()), int(pos.Col()))
}

func lineColOffset(starts []int, size, line, col int) (int, error) {
	if line < 1 || line > len(starts) || col < 1 {
		return 0, fmt.Errorf("position %d:%d is outside the source", line, col)
	}
	end := size
	if line < len(starts) {
		end = starts[line] - 1
	}
	off := starts[line-1] + col - 1
	if off > end {
		return 0, fmt.Errorf("position %d:%d is outside the source", line, col)
	}
	return off, nil
}

func resolve(src []byte, starts []int, base *syntax.PosBase, e Edit) (splice, error) {
	for _, pos := range []syntax.Pos{e.Start, e.End} {
		if pos.FileBase() != base {
			return splice{}, fmt.Errorf("edit position %v belongs to a different file than %s", pos, base.Filename())
		}
	}
	start, err := offsetOf(starts, len(src), e.Start)
	if err != nil {
		return splice{}, err
	}
	end, err := offsetOf(starts, len(src), e.End)
	if err != nil {
		return splice{}, err
	}
	if end < start {
		return splice{}, fmt.Errorf("edit at %v ends before it starts", e.Start)
	}
	return splice{start: start, end: end, text: e.Text, chunks: e.Chunks}, nil
}

func ApplyEdits(src []byte, base *syntax.PosBase, edits []Edit) ([]byte, error) {
	out, _, _, err := ApplyEditsSegments(src, base, edits)
	return out, err
}

func ApplyEditsSegments(src []byte, base *syntax.PosBase, edits []Edit) ([]byte, []editmap.EditSpan, []editmap.Segment, error) {
	want := base.Pos().FileBase()
	starts := lineStarts(src)
	splices := make([]splice, len(edits))
	for i, e := range edits {
		s, err := resolve(src, starts, want, e)
		if err != nil {
			return nil, nil, nil, err
		}
		splices[i] = s
	}
	text, err := applyChunks(otext{s: string(src)}, splices)
	if err != nil {
		return nil, nil, nil, err
	}
	text.checkChunks(src)
	out := []byte(text.s)
	return out, spansOf(splices), segmentsOf(splices, len(out)), nil
}

func spansOf(sorted []splice) []editmap.EditSpan {
	spans := make([]editmap.EditSpan, len(sorted))
	for i, s := range sorted {
		spans[i] = editmap.EditSpan{Start: s.start, End: s.end, Len: len(s.text)}
	}
	return spans
}

func segmentsOf(sorted []splice, outLen int) []editmap.Segment {
	var segs []editmap.Segment
	seg := func(start, end, orig int) {
		if end <= start {
			return
		}
		if n := len(segs); n > 0 {
			last := &segs[n-1]
			if (last.Orig < 0 && orig < 0) ||
				(last.Orig >= 0 && orig >= 0 && last.Orig+last.Len == orig && last.Out+last.Len == start) {
				last.Len += end - start
				return
			}
		}
		segs = append(segs, editmap.Segment{Out: start, Len: end - start, Orig: orig})
	}
	out := 0
	prev := 0
	for _, s := range sorted {
		seg(out, out+(s.start-prev), prev)
		out += s.start - prev
		cov := 0
		for _, ch := range s.chunks {
			if ch.n == 0 {
				continue
			}
			if ch.out > cov {
				seg(out+cov, out+ch.out, -1)
			}
			seg(out+ch.out, out+ch.out+ch.n, ch.orig)
			cov = ch.out + ch.n
		}
		if cov < len(s.text) {
			seg(out+cov, out+len(s.text), -1)
		}
		out += len(s.text)
		prev = s.end
	}
	seg(out, outLen, prev)
	return segs
}

func sortSplices(splices []splice) {
	slices.SortStableFunc(splices, func(a, b splice) int {
		return cmp.Or(cmp.Compare(a.start, b.start), cmp.Compare(a.end, b.end), cmp.Compare(a.placement, b.placement))
	})
}

func applySplices(src []byte, splices []splice) ([]byte, error) {
	t, err := applyChunks(otext{s: string(src)}, splices)
	return []byte(t.s), err
}

func applyChunks(src otext, splices []splice) (otext, error) {
	sortSplices(splices)
	var sb strings.Builder
	var chunks []chunk
	out := 0
	prev := 0
	gap := func(upto int) {
		if upto <= prev {
			return
		}
		sb.WriteString(src.s[prev:upto])
		for _, c := range src.slice(prev, upto).chunks {
			chunks = append(chunks, chunk{out: out + c.out, orig: c.orig, n: c.n})
		}
		out += upto - prev
	}
	var last splice
	for _, s := range splices {
		if s.start < prev {
			return otext{}, fmt.Errorf("overlapping edits at byte offset %d: [%d,%d)->%q, [%d,%d)->%q", s.start, last.start, last.end, last.text, s.start, s.end, s.text)
		}
		gap(s.start)
		sb.WriteString(s.text)
		for _, c := range s.chunks {
			if c.n == 0 {
				continue
			}
			chunks = append(chunks, chunk{out: out + c.out, orig: c.orig, n: c.n})
		}
		out += len(s.text)
		prev = s.end
		last = s
	}
	gap(len(src.s))
	res := otext{s: sb.String(), chunks: chunks}
	if res.s == "" {
		return otext{}, nil
	}
	return res, nil
}

func editsOf(src []byte, base *syntax.PosBase, splices []splice) []Edit {
	starts := lineStarts(src)
	posAt := func(off int) syntax.Pos {
		line := sort.SearchInts(starts, off+1) - 1
		return syntax.MakePos(base, uint(line+1), uint(off-starts[line]+1))
	}
	sorted := slices.Clone(splices)
	sortSplices(sorted)
	edits := make([]Edit, len(sorted))
	for i, s := range sorted {
		edits[i] = Edit{Start: posAt(s.start), End: posAt(s.end), Text: s.text, Chunks: s.chunks}
	}
	return edits
}

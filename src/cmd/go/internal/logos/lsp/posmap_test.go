// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"bytes"
	"math/rand/v2"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"cmd/internal/logos/editmap"
)

func questionMap(t *testing.T) (src, out string, pm *posMap) {
	t.Helper()
	lowered, m, ok := lowerQuestion([]byte(aLgo))
	if !ok {
		t.Fatal("aLgo does not lower")
	}
	m.Source = "a.lgo"
	pm, err := newPosMap([]byte(aLgo), lowered, m)
	if err != nil {
		t.Fatal(err)
	}
	return aLgo, string(lowered), pm
}

type segBuilder struct {
	pos  int
	segs []editmap.Segment
}

func (b *segBuilder) copy(orig, n int) {
	if n <= 0 {
		return
	}
	if k := len(b.segs); k > 0 {
		if last := &b.segs[k-1]; last.Orig >= 0 && last.Orig+last.Len == orig && last.Out+last.Len == b.pos {
			last.Len += n
			b.pos += n
			return
		}
	}
	b.segs = append(b.segs, editmap.Segment{Out: b.pos, Len: n, Orig: orig})
	b.pos += n
}

func (b *segBuilder) gen(n int) {
	if n <= 0 {
		return
	}
	if k := len(b.segs); k > 0 {
		if last := &b.segs[k-1]; last.Orig == -1 && last.Out+last.Len == b.pos {
			last.Len += n
			b.pos += n
			return
		}
	}
	b.segs = append(b.segs, editmap.Segment{Out: b.pos, Len: n, Orig: -1})
	b.pos += n
}

func (b *segBuilder) done() []editmap.Segment {
	return b.segs
}

func spanSegs(srcLen int, edits []editmap.EditSpan) []editmap.Segment {
	var b segBuilder
	prev := 0
	for _, s := range edits {
		b.copy(prev, s.Start-prev)
		b.gen(s.Len)
		prev = s.End
	}
	b.copy(prev, srcLen-prev)
	return b.done()
}

func TestPosMapKeepsOperandOfQuestionStatement(t *testing.T) {
	src, out, pm := questionMap(t)
	srcLine := lineOf(src, 5)
	for _, word := range []string{"data", "os", "ReadFile", "path"} {
		orig := Position{Line: 5, Character: strings.Index(srcLine, word)}
		low := pm.toLowered(orig)
		if got := lineOf(out, low.Line)[low.Character:]; !strings.HasPrefix(got, word) {
			t.Errorf("%s: source %v maps to lowered %v, which starts %q", word, orig, low, got)
		}
		if back := pm.toOriginal(low); back != orig {
			t.Errorf("%s: round trip %v -> %v -> %v", word, orig, low, back)
		}
	}
}

func TestPosMapRangeToLoweredCoversQuestionStatement(t *testing.T) {
	src, out, pm := questionMap(t)
	srcLine := lineOf(src, 5)
	r := Range{
		Start: Position{Line: 5, Character: strings.Index(srcLine, "data")},
		End:   Position{Line: 5, Character: len(srcLine)},
	}
	got := pm.rangeToLowered(r)
	if got.Start != (Position{Line: 5, Character: 1}) || got.End != (Position{Line: 7, Character: len(lineOf(out, 7))}) {
		t.Errorf("range %v maps to %v, want 5:1 to the end of lowered line 7", r, got)
	}
}

func TestPosMapInsertionAtEndOfFile(t *testing.T) {
	_, out, pm := questionMap(t)
	end := len(aLgo)
	if got, ok := pm.LoweredOffset(end, editmap.Trailing); !ok || got != len(out)-len(siteTrailer) {
		t.Errorf("trailing end maps to (%d, %v), want (%d, true)", got, ok, len(out)-len(siteTrailer))
	}
	if got, ok := pm.LoweredOffset(end, editmap.Leading); !ok || got != len(out) {
		t.Errorf("leading end maps to (%d, %v), want (%d, true)", got, ok, len(out))
	}
	if got, gen := pm.OriginalOffset(len(out), editmap.Leading); gen || got != end {
		t.Errorf("end of lowered text maps to (%d, %v), want (%d, false)", got, gen, end)
	}
}

func TestPosMapDeletion(t *testing.T) {
	const deleted = "import \"comptime\"\n"
	src := []byte(deleted + "x := 1\n")
	out := []byte("x := 1\n")
	edits := []editmap.EditSpan{{Start: 0, End: len(deleted), Len: 0}}
	pm, err := newPosMap(src, out, editmap.EditMap{
		Version: editmap.Version, Source: "d.lgo", Size: len(src), Out: len(out),
		Edits: edits, Segments: spanSegs(len(src), edits),
	})
	if err != nil {
		t.Fatal(err)
	}
	if got, gen := pm.OriginalOffset(0, editmap.Leading); gen || got != len(deleted) {
		t.Errorf("leading seam maps to (%d, %v), want (%d, false) after the deleted line", got, gen, len(deleted))
	}
	if got, gen := pm.OriginalOffset(0, editmap.Trailing); gen || got != 0 {
		t.Errorf("trailing seam maps to (%d, %v), want (0, false) before the deleted line", got, gen)
	}
	if got := pm.toLowered(Position{Line: 1, Character: 0}); got != (Position{Line: 0, Character: 0}) {
		t.Errorf("the line after the deletion maps to %v, want 0:0", got)
	}
}

func TestNewPosMapRejectsBadMaps(t *testing.T) {
	src, out := []byte("abc"), []byte("aXXc")
	goodSegs := spanSegs(len(src), []editmap.EditSpan{{Start: 1, End: 2, Len: 2}})
	good := editmap.EditMap{Version: editmap.Version, Source: "x.lgo", Size: 3, Out: 4, Edits: []editmap.EditSpan{{Start: 1, End: 2, Len: 2}}, Segments: goodSegs}
	if _, err := newPosMap(src, out, good); err != nil {
		t.Fatalf("good map rejected: %v", err)
	}
	bad := func(size, outSize int, edits ...editmap.EditSpan) editmap.EditMap {
		return editmap.EditMap{Version: editmap.Version, Source: "x.lgo", Size: size, Out: outSize, Edits: edits, Segments: goodSegs}
	}
	cases := map[string]editmap.EditMap{
		"version":      {Version: 1, Source: "x.lgo", Size: 3, Out: 4, Edits: good.Edits, Segments: goodSegs},
		"source size":  bad(4, 4, good.Edits...),
		"out size":     bad(3, 5, good.Edits...),
		"sum":          bad(3, 4, editmap.EditSpan{Start: 1, End: 2, Len: 3}),
		"past the end": bad(3, 4, editmap.EditSpan{Start: 1, End: 9, Len: 10}),
		"inverted":     bad(3, 4, editmap.EditSpan{Start: 2, End: 1, Len: 3}),
		"negative len": bad(3, 4, editmap.EditSpan{Start: 1, End: 2, Len: -1}, editmap.EditSpan{Start: 2, End: 2, Len: 2}),
		"overlap":      bad(3, 5, editmap.EditSpan{Start: 0, End: 2, Len: 2}, editmap.EditSpan{Start: 1, End: 2, Len: 2}),
		"out of order": bad(3, 4, editmap.EditSpan{Start: 2, End: 3, Len: 1}, editmap.EditSpan{Start: 0, End: 1, Len: 1}),
		"no segments": {
			Version: editmap.Version, Source: "x.lgo", Size: 3, Out: 4,
			Edits: good.Edits,
		},
		"segments short": {
			Version: editmap.Version, Source: "x.lgo", Size: 3, Out: 4,
			Edits:    good.Edits,
			Segments: []editmap.Segment{{Out: 0, Len: 3, Orig: 0}},
		},
		"segments overlap": {
			Version: editmap.Version, Source: "x.lgo", Size: 3, Out: 4,
			Edits:    good.Edits,
			Segments: []editmap.Segment{{Out: 0, Len: 2, Orig: -1}, {Out: 1, Len: 3, Orig: -1}},
		},
		"copy outside source": {
			Version: editmap.Version, Source: "x.lgo", Size: 3, Out: 4,
			Edits:    good.Edits,
			Segments: []editmap.Segment{{Out: 0, Len: 4, Orig: 3}},
		},
		"copy bytes differ": {
			Version: editmap.Version, Source: "x.lgo", Size: 3, Out: 4,
			Edits:    good.Edits,
			Segments: []editmap.Segment{{Out: 0, Len: 1, Orig: 0}, {Out: 1, Len: 2, Orig: -1}, {Out: 3, Len: 1, Orig: 0}},
		},
	}
	for name, m := range cases {
		if _, err := newPosMap(src, out, m); err == nil {
			t.Errorf("%s: map %+v accepted", name, m)
		}
	}
}

func TestPosMapSegmentsMapCopiedRunsExactly(t *testing.T) {
	_, _, pm := questionMap(t)
	copied := 0
	for _, s := range pm.Segments {
		if s.Orig < 0 {
			continue
		}
		copied += s.Len
		for i := range s.Len {
			if got, ok := pm.LoweredOffset(s.Orig+i, editmap.Leading); !ok || got != s.Out+i {
				t.Errorf("copied original %d maps to (%d, %v), want (%d, true)", s.Orig+i, got, ok, s.Out+i)
			}
			if i > 0 {
				if got, ok := pm.LoweredOffset(s.Orig+i, editmap.Trailing); !ok || got != s.Out+i {
					t.Errorf("copied original %d trails to (%d, %v), want (%d, true)", s.Orig+i, got, ok, s.Out+i)
				}
			}
			if back, gen := pm.OriginalOffset(s.Out+i, editmap.Leading); gen || back != s.Orig+i {
				t.Errorf("copied lowered %d maps to (%d, %v), want (%d, false)", s.Out+i, back, gen, s.Orig+i)
			}
		}
	}
	if copied == 0 {
		t.Error("the ? lowering carries no copied bytes")
	}
}

func TestPosMapLargeReplacementMapsBySegments(t *testing.T) {
	src := []byte("w" + strings.Repeat("(", 2048))
	mid := src[512:1536]
	out := append(append([]byte("gen<"), mid...), []byte(">gen")...)
	edits := []editmap.EditSpan{{Start: 0, End: len(src), Len: len(out)}}
	m := editmap.EditMap{
		Version: editmap.Version, Source: "big.lgo", Size: len(src), Out: len(out),
		Edits: edits,
		Segments: []editmap.Segment{
			{Out: 0, Len: 4, Orig: -1},
			{Out: 4, Len: len(mid), Orig: 512},
			{Out: 4 + len(mid), Len: 4, Orig: -1},
		},
	}
	pm, err := newPosMap(src, out, m)
	if err != nil {
		t.Fatal(err)
	}
	for i := range mid {
		if got, ok := pm.LoweredOffset(512+i, editmap.Leading); !ok || got != 4+i {
			t.Fatalf("copied original %d maps to (%d, %v), want (%d, true)", 512+i, got, ok, 4+i)
		}
		if back, gen := pm.OriginalOffset(4+i, editmap.Leading); gen || back != 512+i {
			t.Fatalf("copied lowered %d maps to (%d, %v), want (%d, false)", 4+i, back, gen, 512+i)
		}
	}
	if got, ok := pm.LoweredOffset(0, editmap.Leading); ok || got != 0 {
		t.Errorf("deleted original 0 maps to (%d, %v), want (0, false)", got, ok)
	}
	if got, gen := pm.OriginalOffset(1, editmap.Leading); !gen || got != 0 {
		t.Errorf("generated prefix maps to (%d, %v), want (0, true)", got, gen)
	}
	if got, gen := pm.OriginalOffset(len(out)-2, editmap.Leading); !gen || got != 1536 {
		t.Errorf("generated suffix maps to (%d, %v), want (1536, true)", got, gen)
	}
}

func assertSpansConsistent(t *testing.T, src, out []byte, spans []editmap.EditSpan) {
	t.Helper()
	prev, pos := 0, 0
	for i, s := range spans {
		if s.Start < prev || s.End < s.Start || s.End > len(src) || s.Len < 0 {
			t.Fatalf("span %d %+v is out of order or out of range", i, s)
		}
		gap := src[prev:s.Start]
		if pos > len(out) || !bytes.HasPrefix(out[pos:], gap) {
			t.Fatalf("source bytes %d:%d do not appear unchanged at lowered offset %d", prev, s.Start, pos)
		}
		pos += len(gap) + s.Len
		prev = s.End
	}
	if pos > len(out) || !bytes.Equal(src[prev:], out[pos:]) {
		t.Fatalf("the source tail from %d does not match the lowered tail from %d", prev, pos)
	}
}

func randomText(rng *rand.Rand, n int) string {
	pieces := []string{"a", "bc", "data", "os", "path", "x1", ".", "(", ")", ":=", ",", " ", " ", "\n", "\t", "é", "𝒳"}
	var b strings.Builder
	for range n {
		b.WriteString(pieces[rng.IntN(len(pieces))])
	}
	return b.String()
}

func randomLowering(rng *rand.Rand, src string) ([]byte, editmap.EditMap) {
	var cuts []int
	for range 1 + rng.IntN(5) {
		cuts = append(cuts, rng.IntN(len(src)+1), rng.IntN(len(src)+1))
	}
	slices.Sort(cuts)
	var out strings.Builder
	var spans []editmap.EditSpan
	var segs segBuilder
	prev := 0
	for i := 0; i+1 < len(cuts); i += 2 {
		start, end := cuts[i], cuts[i+1]
		pre := randomText(rng, rng.IntN(5))
		post := randomText(rng, rng.IntN(5))
		out.WriteString(src[prev:start])
		segs.copy(prev, start-prev)
		if rng.IntN(3) == 0 {
			repl := randomText(rng, rng.IntN(6))
			out.WriteString(repl)
			segs.gen(len(repl))
			spans = append(spans, editmap.EditSpan{Start: start, End: end, Len: len(repl)})
		} else {
			out.WriteString(pre)
			segs.gen(len(pre))
			out.WriteString(src[start:end])
			segs.copy(start, end-start)
			out.WriteString(post)
			segs.gen(len(post))
			spans = append(spans, editmap.EditSpan{Start: start, End: end, Len: len(pre) + (end - start) + len(post)})
		}
		prev = end
	}
	out.WriteString(src[prev:])
	segs.copy(prev, len(src)-prev)
	return []byte(out.String()), editmap.EditMap{Version: editmap.Version, Source: "r.lgo", Size: len(src), Out: out.Len(), Edits: spans, Segments: segs.done()}
}

func TestPosMapProperties(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	for iter := range 400 {
		src := randomText(rng, 3+rng.IntN(60))
		out, m := randomLowering(rng, src)
		pm, err := newPosMap([]byte(src), out, m)
		if err != nil {
			t.Fatalf("iteration %d: %v\nsource %q\nmap %+v", iter, err, src, m)
		}
		assertSpansConsistent(t, []byte(src), out, m.Edits)
		checkProperties(t, iter, []byte(src), out, pm)
	}
}

func checkProperties(t *testing.T, iter int, src, out []byte, pm *posMap) {
	t.Helper()
	for _, s := range pm.Segments {
		if s.Orig < 0 {
			continue
		}
		for i := range s.Len {
			if out[s.Out+i] != src[s.Orig+i] {
				t.Fatalf("iteration %d: copied lowered %d (%q) differs from original %d (%q)", iter, s.Out+i, out[s.Out+i], s.Orig+i, src[s.Orig+i])
			}
			if back, gen := pm.OriginalOffset(s.Out+i, editmap.Leading); gen || back != s.Orig+i {
				t.Fatalf("iteration %d: lowered %d maps to (%d, %v), want (%d, false)", iter, s.Out+i, back, gen, s.Orig+i)
			}
		}
		start, _ := pm.OriginalOffset(s.Out, editmap.Leading)
		if start != s.Orig {
			t.Fatalf("iteration %d: run at %d maps to %d, want %d", iter, s.Out, start, s.Orig)
		}
		finish, _ := pm.OriginalOffset(s.Out+s.Len, editmap.Trailing)
		if finish != s.Orig+s.Len {
			t.Fatalf("iteration %d: run end at %d maps to %d, want %d", iter, s.Out+s.Len, finish, s.Orig+s.Len)
		}
		if pm.low.offset(pm.low.position(s.Out)) != s.Out {
			continue
		}
		if pm.low.offset(pm.low.position(s.Out+s.Len)) != s.Out+s.Len {
			continue
		}
		if pm.orig.offset(pm.orig.position(s.Orig)) != s.Orig {
			continue
		}
		if pm.orig.offset(pm.orig.position(s.Orig+s.Len)) != s.Orig+s.Len {
			continue
		}
		got := pm.rangeToOriginal(Range{Start: pm.low.position(s.Out), End: pm.low.position(s.Out + s.Len)})
		want := Range{Start: pm.orig.position(s.Orig), End: pm.orig.position(s.Orig + s.Len)}
		if got != want {
			t.Fatalf("iteration %d: lowered range %v maps to %v, want %v", iter, s, got, want)
		}
	}
	seen := make(map[int]struct{})
	for _, s := range pm.Segments {
		if s.Orig < 0 {
			continue
		}
		for i := range s.Len {
			o := s.Orig + i
			if _, ok := seen[o]; ok {
				continue
			}
			seen[o] = struct{}{}
			low, ok := pm.LoweredOffset(o, editmap.Leading)
			if !ok {
				t.Fatalf("iteration %d: copied original %d has no lowered position", iter, o)
			}
			if back, gen := pm.OriginalOffset(low, editmap.Leading); gen || back != o {
				t.Fatalf("iteration %d: original %d -> %d -> (%d, %v) is not the identity", iter, o, low, back, gen)
			}
		}
	}
}

func TestPosMapMovedDuplicatedExactChunks(t *testing.T) {
	src := []byte("ABCDE FGHIJ")
	movedOut := append(append([]byte("FGHIJ "), []byte("ABCDE")...), []byte("!")...)
	moved := editmap.EditMap{
		Version: editmap.Version, Source: "m.lgo", Size: len(src), Out: len(movedOut),
		Edits: []editmap.EditSpan{{Start: 0, End: len(src), Len: len(movedOut)}},
		Segments: []editmap.Segment{
			{Out: 0, Len: 5, Orig: 6},
			{Out: 5, Len: 1, Orig: -1},
			{Out: 6, Len: 5, Orig: 0},
			{Out: 11, Len: 1, Orig: -1},
		},
	}
	pm, err := newPosMap(src, movedOut, moved)
	if err != nil {
		t.Fatal(err)
	}
	checkProperties(t, 0, src, movedOut, pm)

	dupOut := []byte("AB AB")
	dup := editmap.EditMap{
		Version: editmap.Version, Source: "d.lgo", Size: 2, Out: len(dupOut),
		Edits: []editmap.EditSpan{{Start: 0, End: 0, Len: 1}, {Start: 2, End: 2, Len: 2}},
		Segments: []editmap.Segment{
			{Out: 0, Len: 2, Orig: 0},
			{Out: 2, Len: 1, Orig: -1},
			{Out: 3, Len: 2, Orig: 0},
		},
	}
	dpm, err := newPosMap([]byte("AB"), dupOut, dup)
	if err != nil {
		t.Fatal(err)
	}
	checkProperties(t, 1, []byte("AB"), dupOut, dpm)
}

func TestTextOffsetsUseUTF16(t *testing.T) {
	const doc = "a𝒳é\nxyz\r\nlast"
	tx := newText([]byte(doc))
	cases := []struct {
		off int
		pos Position
	}{
		{0, Position{0, 0}},
		{1, Position{0, 1}},
		{5, Position{0, 3}},
		{7, Position{0, 4}},
		{8, Position{1, 0}},
		{11, Position{1, 3}},
		{13, Position{2, 0}},
	}
	for _, c := range cases {
		if got := tx.position(c.off); got != c.pos {
			t.Errorf("position(%d) = %v, want %v", c.off, got, c.pos)
		}
		if got := tx.offset(c.pos); got != c.off {
			t.Errorf("offset(%v) = %d, want %d", c.pos, got, c.off)
		}
	}
	if got := tx.offset(Position{Line: 0, Character: 99}); got != 7 {
		t.Errorf("a character past the line end gives %d, want the line end 7", got)
	}
	if got := tx.offset(Position{Line: 99}); got != len(doc) {
		t.Errorf("a line past the end gives %d, want the text size", got)
	}
	if off, ok := tx.lineCol(2, 4); !ok || off != 11 {
		t.Errorf("lineCol(2, 4) = %d, %v, want 11, true", off, ok)
	}
	if _, ok := tx.lineCol(2, 40); ok {
		t.Error("lineCol accepted a column past the end of its line")
	}
}

func TestApplyChangesIncremental(t *testing.T) {
	tx := newText([]byte("héllo\nworld\n"))
	tx = applyChanges(tx, []contentChange{
		{Range: &Range{Start: Position{0, 1}, End: Position{0, 2}}, Text: "e"},
		{Range: &Range{Start: Position{1, 0}, End: Position{1, 0}}, Text: ">> "},
	})
	if got := string(tx.data); got != "hello\n>> world\n" {
		t.Errorf("after range changes: %q", got)
	}
	tx = applyChanges(tx, []contentChange{{Text: "new"}})
	if got := string(tx.data); got != "new" {
		t.Errorf("after whole-text change: %q", got)
	}
}

func mustPosMap(t *testing.T, src, out string, m editmap.EditMap) *posMap {
	t.Helper()
	pm, err := newPosMap([]byte(src), []byte(out), m)
	if err != nil {
		t.Fatalf("newPosMap(%q, %q): %v", src, out, err)
	}
	return pm
}

func TestPosMapGeneratedPrefixAtZeroAndEmptySource(t *testing.T) {
	prefix := mustPosMap(t, "ab\n", "PRE\nab\n", editmap.EditMap{
		Version: editmap.Version, Source: "p.lgo", Size: 3, Out: 7,
		Edits:    []editmap.EditSpan{{Start: 0, End: 0, Len: 4}},
		Segments: spanSegs(3, []editmap.EditSpan{{Start: 0, End: 0, Len: 4}}),
	})
	if got := prefix.toLowered(Position{Line: 0, Character: 0}); got != (Position{Line: 1, Character: 0}) {
		t.Errorf("source start maps to %v, want the first copied line 1:0", got)
	}
	if got := prefix.toLowered(Position{Line: 0, Character: 1}); got != (Position{Line: 1, Character: 1}) {
		t.Errorf("source 0:1 maps to %v, want lowered 1:1", got)
	}
	if got := prefix.toOriginal(Position{Line: 0, Character: 1}); got != (Position{Line: 0, Character: 0}) {
		t.Errorf("a position inside the prefix maps to %v, want the source start", got)
	}
	if got := prefix.rangeToLowered(Range{Start: Position{Line: 0, Character: 0}, End: Position{Line: 0, Character: 2}}); got != (Range{Start: Position{Line: 1, Character: 0}, End: Position{Line: 1, Character: 2}}) {
		t.Errorf("source line range maps to %v, want lowered 1:0-1:2", got)
	}
	if got := prefix.rangeToOriginal(Range{Start: Position{Line: 0, Character: 0}, End: Position{Line: 0, Character: 3}}); got != (Range{Start: Position{}, End: Position{}}) {
		t.Errorf("a range over only the prefix maps to %v, want the empty range at the source start", got)
	}
	if got, ok := prefix.LoweredOffset(0, editmap.Trailing); !ok || got != 0 {
		t.Errorf("trailing offset zero maps to (%d, %v), want (0, true) before the prefix", got, ok)
	}

	empty := mustPosMap(t, "", "GEN", editmap.EditMap{
		Version: editmap.Version, Source: "e.lgo", Size: 0, Out: 3,
		Edits:    []editmap.EditSpan{{Start: 0, End: 0, Len: 3}},
		Segments: spanSegs(0, []editmap.EditSpan{{Start: 0, End: 0, Len: 3}}),
	})
	if got := empty.toLowered(Position{}); got != (Position{Line: 0, Character: 3}) {
		t.Errorf("the empty source maps to %v, want the end of the generated output", got)
	}
	if got := empty.toOriginal(Position{Line: 0, Character: 1}); got != (Position{}) {
		t.Errorf("generated output of an empty source maps to %v, want the source start", got)
	}
	if got := empty.rangeToOriginal(Range{Start: Position{}, End: Position{Line: 0, Character: 3}}); got != (Range{}) {
		t.Errorf("the whole generated output maps to %v, want the empty source range", got)
	}
}

func TestPosMapMidTextInsertionSeams(t *testing.T) {
	edits := []editmap.EditSpan{{Start: 1, End: 1, Len: 3}}
	pm := mustPosMap(t, "ab", "aXYZb", editmap.EditMap{
		Version: editmap.Version, Source: "m.lgo", Size: 2, Out: 5,
		Edits: edits, Segments: spanSegs(2, edits),
	})
	if got := pm.toLowered(Position{Line: 0, Character: 1}); got != (Position{Line: 0, Character: 4}) {
		t.Errorf("the seam maps to %v, want lowered 0:4 after the insertion", got)
	}
	if got := pm.rangeToLowered(Range{Start: Position{Line: 0, Character: 0}, End: Position{Line: 0, Character: 1}}); got != (Range{Start: Position{Line: 0, Character: 0}, End: Position{Line: 0, Character: 1}}) {
		t.Errorf("the range before the insertion maps to %v, want itself", got)
	}
	if got := pm.rangeToLowered(Range{Start: Position{Line: 0, Character: 1}, End: Position{Line: 0, Character: 2}}); got != (Range{Start: Position{Line: 0, Character: 4}, End: Position{Line: 0, Character: 5}}) {
		t.Errorf("the range after the insertion maps to %v, want lowered 0:4-0:5", got)
	}
	if got := pm.toOriginal(Position{Line: 0, Character: 2}); got != (Position{Line: 0, Character: 1}) {
		t.Errorf("inserted bytes map to %v, want the insertion point 0:1", got)
	}
}

func TestPosMapMidAndEndTextDeletionNeighbors(t *testing.T) {
	mid := mustPosMap(t, "one\ndropped\ntwo\n", "one\ntwo\n", editmap.EditMap{
		Version: editmap.Version, Source: "d.lgo", Size: 16, Out: 8,
		Edits:    []editmap.EditSpan{{Start: 4, End: 12, Len: 0}},
		Segments: spanSegs(16, []editmap.EditSpan{{Start: 4, End: 12, Len: 0}}),
	})
	if got := mid.toLowered(Position{Line: 2, Character: 0}); got != (Position{Line: 1, Character: 0}) {
		t.Errorf("the line after the deletion maps to %v, want lowered 1:0", got)
	}
	if got := mid.toLowered(Position{Line: 1, Character: 0}); got != (Position{Line: 1, Character: 0}) {
		t.Errorf("a position inside deleted text maps to %v, want the start of its replacement", got)
	}
	if got := mid.toOriginal(Position{Line: 1, Character: 0}); got != (Position{Line: 2, Character: 0}) {
		t.Errorf("the seam maps to %v, want the original line after the deletion", got)
	}
	if got := mid.rangeToOriginal(Range{Start: Position{Line: 0, Character: 0}, End: Position{Line: 1, Character: 3}}); got != (Range{Start: Position{Line: 0, Character: 0}, End: Position{Line: 2, Character: 3}}) {
		t.Errorf("a range across the deletion maps to %v, want the original 0:0-2:3", got)
	}

	tail := mustPosMap(t, "keep\ngone\n", "keep\n", editmap.EditMap{
		Version: editmap.Version, Source: "t.lgo", Size: 10, Out: 5,
		Edits:    []editmap.EditSpan{{Start: 5, End: 10, Len: 0}},
		Segments: spanSegs(10, []editmap.EditSpan{{Start: 5, End: 10, Len: 0}}),
	})
	if got := tail.toLowered(Position{Line: 1, Character: 3}); got != (Position{Line: 1, Character: 0}) {
		t.Errorf("a position inside the deleted tail maps to %v, want the lowered end", got)
	}
	if got := tail.toOriginal(Position{Line: 1, Character: 0}); got != (Position{Line: 2, Character: 0}) {
		t.Errorf("the lowered end maps to %v, want the original end past the deleted tail", got)
	}
	if got := tail.rangeToOriginal(Range{Start: Position{Line: 0, Character: 0}, End: Position{Line: 1, Character: 0}}); got != (Range{Start: Position{Line: 0, Character: 0}, End: Position{Line: 1, Character: 0}}) {
		t.Errorf("the whole lowered text maps to %v, want the original without its deleted tail", got)
	}
}

func TestNewPosMapRejectsDegenerateSegments(t *testing.T) {
	src, out := []byte("abc"), []byte("aXXc")
	edits := []editmap.EditSpan{{Start: 1, End: 2, Len: 2}}
	segs := spanSegs(len(src), edits)
	cases := map[string][]editmap.Segment{
		"zero length": {
			{Out: 0, Len: 1, Orig: 0}, {Out: 1, Len: 0, Orig: -1}, {Out: 1, Len: 2, Orig: -1}, {Out: 3, Len: 1, Orig: 2},
		},
		"origin below generated": {
			{Out: 0, Len: 1, Orig: 0}, {Out: 1, Len: 2, Orig: -2}, {Out: 3, Len: 1, Orig: 2},
		},
		"gap at start": {
			{Out: 1, Len: 2, Orig: -1}, {Out: 3, Len: 1, Orig: 2},
		},
		"run past the text": {
			{Out: 0, Len: 1, Orig: 0}, {Out: 1, Len: 4, Orig: -1},
		},
	}
	for name, s := range cases {
		m := editmap.EditMap{Version: editmap.Version, Source: "x.lgo", Size: 3, Out: 4, Edits: edits, Segments: s}
		if _, err := newPosMap(src, out, m); err == nil {
			t.Errorf("%s: degenerate map accepted", name)
		}
	}
	if _, err := newPosMap(src, out, editmap.EditMap{Version: editmap.Version, Source: "x.lgo", Size: 3, Out: 4, Edits: edits, Segments: segs}); err != nil {
		t.Errorf("the control map is rejected: %v", err)
	}
}

const rewrittenLgo = `package z

import (
	"fmt"
	"strconv"
)

func Parse(s string) (int, error) {
	n := strconv.Atoi(s)?
	return n, nil
}

func Double(xs []int) []int {
	return xs.map(x => x * 2)
}

func SumTwice(xs []int) []int {
	return xs.map(x => {
		return x * 2
	})
}

func First(xs []int) (string, error) {
	ys := xs.map(x => Parse(fmt.Sprint(x))?)
	return fmt.Sprint(len(ys)), nil
}
`

func TestPosMapRealLoweringKeepsRewrittenCoordinates(t *testing.T) {
	root, tc := realWorkspace(t)
	zdir := filepath.Join(root, "lmod", "z")
	if err := os.MkdirAll(zdir, 0o777); err != nil {
		t.Fatal(err)
	}
	zpath := filepath.Join(zdir, "z.lgo")
	if err := os.WriteFile(zpath, []byte(rewrittenLgo), 0o666); err != nil {
		t.Fatal(err)
	}
	low, err := tc.Lower(t.Context(), zdir, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(low.Missing) != 0 || len(low.Files) != 1 || low.Files[0].Path != zpath {
		t.Fatalf("lowering = %+v, want the one file %s", low, zpath)
	}
	file := low.Files[0]
	out := string(file.Out)
	if strings.Contains(out, "?") {
		t.Errorf("lowered text keeps a ?:\n%s", out)
	}
	pm, err := newPosMap(file.Src, file.Out, file.Map)
	if err != nil {
		t.Fatalf("the compiler's edit map does not fit its texts: %v", err)
	}
	needles := []struct {
		text string
		nth  int
	}{
		{"strconv.Atoi(s)", 0},
		{"x * 2", 0},
		{"x * 2", 1},
		{"Parse(fmt.Sprint(x))", 0},
		{"len(ys)", 0},
	}
	for _, n := range needles {
		orig := findPos(t, rewrittenLgo, n.text, n.nth)
		lowPos := pm.toLowered(orig)
		if got := lineOf(out, lowPos.Line)[lowPos.Character:]; !strings.HasPrefix(got, n.text) {
			t.Errorf("%q maps to lowered %v, which starts %q", n.text, lowPos, got)
		}
		if back := pm.toOriginal(lowPos); back != orig {
			t.Errorf("%q round trip %v -> %v -> %v", n.text, orig, lowPos, back)
		}
	}
	checkProperties(t, 0, file.Src, file.Out, pm)
	generated := 0
	for _, s := range pm.Segments {
		if s.Orig >= 0 {
			continue
		}
		generated += s.Len
		if s.Len >= 2 {
			if _, gen := pm.OriginalOffset(s.Out+1, editmap.Leading); !gen {
				t.Errorf("inside generated run %v maps as user text", s)
			}
		}
	}
	if generated == 0 {
		t.Error("the lowering carries no generated bytes")
	}
	count := 0
	for i := strings.Index(out, "_logos_"); i >= 0; {
		count++
		if _, gen := pm.OriginalOffset(i+1, editmap.Leading); !gen {
			t.Errorf("invented identifier at lowered offset %d maps as user text", i)
		}
		next := strings.Index(out[i+1:], "_logos_")
		if next < 0 {
			break
		}
		i += 1 + next
	}
	if count == 0 {
		t.Error("the lowering invents no _logos_ identifier")
	}
}

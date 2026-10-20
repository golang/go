// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"bytes"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"
)

// questionMap lowers aLgo with the fake lowering and returns the texts and
// the position map between them.
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

// TestPosMapKeepsOperandOfQuestionStatement checks that the operand a ?
// statement copies into its expansion maps exactly, both ways, so
// hover and go-to-definition on `os.ReadFile(path)` hit the same tokens
// in the source and in the lowered text.
func TestPosMapKeepsOperandOfQuestionStatement(t *testing.T) {
	src, out, pm := questionMap(t)
	srcLine := lineOf(src, 5)
	for _, word := range []string{"data", "os", "ReadFile", "path"} {
		orig := Position{Line: 5, Character: strings.Index(srcLine, word)}
		low := pm.toLowered(orig, leading)
		if got := lineOf(out, low.Line)[low.Character:]; !strings.HasPrefix(got, word) {
			t.Errorf("%s: source %v maps to lowered %v, which starts %q", word, orig, low, got)
		}
		if back := pm.toOriginal(low, leading); back != orig {
			t.Errorf("%s: round trip %v -> %v -> %v", word, orig, low, back)
		}
	}
}

// TestPosMapGeneratedLinesMapToQuestionStatement checks that every
// position inside the three lines the ? statement expands to maps back
// into the statement of the source: the first line to its text, the
// generated lines to the ? token that produced them.
func TestPosMapGeneratedLinesMapToQuestionStatement(t *testing.T) {
	src, out, pm := questionMap(t)
	srcLine := lineOf(src, 5)
	qcol := strings.Index(srcLine, "?")
	if qcol < 0 {
		t.Fatal("no ? in the source line")
	}
	for line := 5; line <= 7; line++ {
		text := lineOf(out, line)
		for col := 0; col <= len(text); col++ {
			orig := pm.toOriginal(Position{Line: line, Character: col}, leading)
			if orig.Line != 5 || orig.Character > len(srcLine) {
				t.Errorf("lowered %d:%d maps to %v, outside source line 5", line, col, orig)
			}
			if line > 5 && orig.Character != qcol && orig.Character != qcol+1 {
				t.Errorf("lowered generated %d:%d maps to column %d, want the ? at %d", line, col, orig.Character, qcol)
			}
		}
	}
	q := Position{Line: 5, Character: qcol}
	low := pm.toLowered(q, leading)
	if low.Line != 5 || low.Character != len(lineOf(out, 5)) {
		t.Errorf("? at %v maps to %v, want the end of lowered line 5", q, low)
	}
	rng := pm.rangeToOriginal(Range{Start: Position{Line: 6, Character: 0}, End: Position{Line: 7, Character: len(lineOf(out, 7))}})
	want := Range{Start: q, End: shift(q, 1)}
	if rng != want {
		t.Errorf("range over the generated lines maps to %v, want the ? token %v", rng, want)
	}
}

// TestPosMapRangeToLoweredCoversQuestionStatement checks that a source
// range over the whole statement maps to a range that covers the lowered
// statement, the call first.
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

// TestPosMapInsertionAtEndOfFile checks the edge of text the lowering
// appends: the end of the source maps before the appended text on the
// trailing edge and after it on the leading edge.
func TestPosMapInsertionAtEndOfFile(t *testing.T) {
	_, out, pm := questionMap(t)
	end := len(aLgo)
	if got, want := pm.lowOffset(end, trailing), len(out)-len(siteTrailer); got != want {
		t.Errorf("trailing end maps to %d, want %d", got, want)
	}
	if got := pm.lowOffset(end, leading); got != len(out) {
		t.Errorf("leading end maps to %d, want %d", got, len(out))
	}
	if got := pm.origOffset(len(out), leading); got != end {
		t.Errorf("end of lowered text maps to %d, want %d", got, end)
	}
}

// TestPosMapDeletion checks both edges at the seam of deleted text.
func TestPosMapDeletion(t *testing.T) {
	const deleted = "import \"comptime\"\n"
	src := []byte(deleted + "x := 1\n")
	out := []byte("x := 1\n")
	pm, err := newPosMap(src, out, EditMap{
		Version: EditMapVersion, Source: "d.lgo", Size: len(src), Out: len(out),
		Edits: []EditSpan{{Start: 0, End: len(deleted), Len: 0}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if got := pm.origOffset(0, leading); got != len(deleted) {
		t.Errorf("leading seam maps to %d, want after the deleted line", got)
	}
	if got := pm.origOffset(0, trailing); got != 0 {
		t.Errorf("trailing seam maps to %d, want before the deleted line", got)
	}
	if got := pm.toLowered(Position{Line: 1, Character: 0}, leading); got != (Position{Line: 0, Character: 0}) {
		t.Errorf("the line after the deletion maps to %v, want 0:0", got)
	}
}

// TestNewPosMapRejectsBadMaps checks that a map that cannot describe its
// texts is refused, so a stale or corrupt sidecar never maps positions.
func TestNewPosMapRejectsBadMaps(t *testing.T) {
	src, out := []byte("abc"), []byte("aXXc")
	good := EditMap{Version: EditMapVersion, Source: "x.lgo", Size: 3, Out: 4, Edits: []EditSpan{{Start: 1, End: 2, Len: 2}}}
	if _, err := newPosMap(src, out, good); err != nil {
		t.Fatalf("good map rejected: %v", err)
	}
	bad := func(size, outSize int, edits ...EditSpan) EditMap {
		return EditMap{Version: EditMapVersion, Source: "x.lgo", Size: size, Out: outSize, Edits: edits}
	}
	cases := map[string]EditMap{
		"version":      {Version: 2, Source: "x.lgo", Size: 3, Out: 4, Edits: good.Edits},
		"source size":  bad(4, 4, good.Edits...),
		"out size":     bad(3, 5, good.Edits...),
		"sum":          bad(3, 4, EditSpan{Start: 1, End: 2, Len: 3}),
		"past the end": bad(3, 4, EditSpan{Start: 1, End: 9, Len: 10}),
		"inverted":     bad(3, 4, EditSpan{Start: 2, End: 1, Len: 3}),
		"negative len": bad(3, 4, EditSpan{Start: 1, End: 2, Len: -1}, EditSpan{Start: 2, End: 2, Len: 2}),
		"overlap":      bad(3, 5, EditSpan{Start: 0, End: 2, Len: 2}, EditSpan{Start: 1, End: 2, Len: 2}),
		"out of order": bad(3, 4, EditSpan{Start: 2, End: 3, Len: 1}, EditSpan{Start: 0, End: 1, Len: 1}),
	}
	for name, m := range cases {
		if _, err := newPosMap(src, out, m); err == nil {
			t.Errorf("%s: map %+v accepted", name, m)
		}
	}
}

// TestRefineSpansStaysInsideSpan checks that refinement only replaces a
// span with spans inside it, and that text outside the refined spans is
// byte for byte the same in both texts.
func TestRefineSpansStaysInsideSpan(t *testing.T) {
	src, out, _ := questionMap(t)
	_, m, _ := lowerQuestion([]byte(src))
	refined := refineSpans([]byte(src), []byte(out), m.Edits)
	if len(refined) <= len(m.Edits) {
		t.Fatalf("refinement did not split the ? statement: %d spans before, %d after", len(m.Edits), len(refined))
	}
	assertSpansConsistent(t, []byte(src), []byte(out), refined)
	for _, r := range refined {
		inside := slices.ContainsFunc(m.Edits, func(s EditSpan) bool { return r.Start >= s.Start && r.End <= s.End })
		if !inside {
			t.Errorf("refined span %+v lies outside every original span %+v", r, m.Edits)
		}
	}
}

// TestRefineCellLimit tests the bound on the token table of refinement at
// its edges. A table of maxRefineCells - 1, maxRefineCells, and
// maxRefineCells + 1 cells refines when it does not exceed the bound, and
// a larger one keeps the coarse span of the whole replaced text.
func TestRefineCellLimit(t *testing.T) {
	text := func(tokens int) []byte { return []byte("w" + strings.Repeat("(", tokens-1)) }
	cases := []struct {
		name      string
		origToks  int
		lowToks   int
		wantCells int
		refined   bool
	}{
		{"one cell under the bound", 1022, 1024, maxRefineCells - 1, true},
		{"at the bound", 1023, 1023, maxRefineCells, true},
		{"one cell over the bound", 16, 61680, maxRefineCells + 1, false},
	}
	for _, c := range cases {
		if cells := (c.origToks + 1) * (c.lowToks + 1); cells != c.wantCells {
			t.Fatalf("%s: the case sizes give %d cells, want %d", c.name, cells, c.wantCells)
		}
		orig, low := text(c.origToks), text(c.lowToks)
		parts := refineSpan(EditSpan{Start: 0, End: len(orig), Len: len(low)}, orig, low)
		refined := len(parts) != 1 || parts[0].Start != 0
		if refined != c.refined {
			t.Errorf("%s: refined = %v (%+v), want %v", c.name, refined, parts, c.refined)
		}
	}
}

// assertSpansConsistent checks that spans describe the transformation from
// src to out: ordered, disjoint, inside src, source bytes between them
// unchanged in out, and sizes adding up.
func assertSpansConsistent(t *testing.T, src, out []byte, spans []EditSpan) {
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

// randomText returns n tokens of source-like text: words, punctuation,
// spaces, and newlines.
func randomText(rng *rand.Rand, n int) string {
	pieces := []string{"a", "bc", "data", "os", "path", "x1", ".", "(", ")", ":=", ",", " ", " ", "\n", "\t", "é", "𝒳"}
	var b strings.Builder
	for range n {
		b.WriteString(pieces[rng.IntN(len(pieces))])
	}
	return b.String()
}

// randomLowering applies random non-overlapping edits to src. Replacement
// text is a mix of generated pieces and words copied from the replaced
// text, so refinement has runs to find.
func randomLowering(rng *rand.Rand, src string) ([]byte, EditMap) {
	var cuts []int
	for range 1 + rng.IntN(5) {
		cuts = append(cuts, rng.IntN(len(src)+1), rng.IntN(len(src)+1))
	}
	slices.Sort(cuts)
	var out strings.Builder
	var spans []EditSpan
	prev := 0
	for i := 0; i+1 < len(cuts); i += 2 {
		start, end := cuts[i], cuts[i+1]
		repl := randomText(rng, rng.IntN(5)) + src[start:end] + randomText(rng, rng.IntN(5))
		if rng.IntN(3) == 0 {
			repl = randomText(rng, rng.IntN(6))
		}
		out.WriteString(src[prev:start])
		out.WriteString(repl)
		spans = append(spans, EditSpan{Start: start, End: end, Len: len(repl)})
		prev = end
	}
	out.WriteString(src[prev:])
	return []byte(out.String()), EditMap{Version: EditMapVersion, Source: "r.lgo", Size: len(src), Out: out.Len(), Edits: spans}
}

// TestPosMapProperties is the invariant test of the position map: for
// random sources and random lowerings, text outside the refined spans maps
// byte for byte and back, offsets never run backwards, and an offset
// round trip through the lowered text is the identity outside the spans.
func TestPosMapProperties(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	for iter := range 400 {
		src := randomText(rng, 3+rng.IntN(60))
		out, m := randomLowering(rng, src)
		pm, err := newPosMap([]byte(src), out, m)
		if err != nil {
			t.Fatalf("iteration %d: %v\nsource %q\nmap %+v", iter, err, src, m)
		}
		assertSpansConsistent(t, []byte(src), out, pm.spans)
		checkProperties(t, iter, []byte(src), out, pm)
	}
}

// checkProperties checks the position map invariants of one lowering.
func checkProperties(t *testing.T, iter int, src, out []byte, pm *posMap) {
	t.Helper()
	changed := make([]bool, len(src)+1)
	for _, s := range pm.spans {
		for o := s.Start; o < s.End; o++ {
			changed[o] = true
		}
	}
	prevLead, prevTrail := 0, 0
	for o := range len(src) + 1 {
		lead, trail := pm.lowOffset(o, leading), pm.lowOffset(o, trailing)
		if lead < prevLead || trail < prevTrail || lead > len(out) || trail > len(out) {
			t.Fatalf("iteration %d: offsets run backwards or leave the text at %d: lead %d trail %d", iter, o, lead, trail)
		}
		prevLead, prevTrail = lead, trail
		if o == len(src) || changed[o] {
			continue
		}
		if out[lead] != src[o] {
			t.Fatalf("iteration %d: unchanged byte %d (%q) maps to lowered %d (%q)", iter, o, src[o], lead, out[lead])
		}
		if back := pm.origOffset(lead, leading); back != o {
			t.Fatalf("iteration %d: offset %d -> %d -> %d is not the identity", iter, o, lead, back)
		}
	}
}

// TestTextOffsetsUseUTF16 checks the conversion between byte offsets and
// UTF-16 positions: multi-byte characters count one unit, characters
// beyond the basic plane count two, and a character past a line's end
// clamps to it.
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

// TestApplyChangesIncremental checks didChange handling: a range change
// splices UTF-16 positions, a whole-text change replaces the text, and
// changes apply in order.
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

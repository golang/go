// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package editmap

import (
	"encoding/json"
	"os"
	"slices"
	"strings"
	"testing"
)

func goodMap(src, out string, edits []EditSpan, segs []Segment) EditMap {
	return EditMap{
		Version:  Version,
		Source:   "x.lgo",
		Size:     len(src),
		Out:      len(out),
		Edits:    edits,
		Segments: segs,
	}
}

func mustNew(t *testing.T, src, out string, m EditMap) *Map {
	t.Helper()
	pm, err := New([]byte(src), []byte(out), m)
	if err != nil {
		t.Fatalf("New(%q, %q, %+v): %v", src, out, m, err)
	}
	return pm
}

func TestWriteRoundTrip(t *testing.T) {
	path := t.TempDir() + "/a.go" + Suffix
	spans := []EditSpan{{Start: 1, End: 2, Len: 2}}
	segs := []Segment{{Out: 0, Len: 1, Orig: 0}, {Out: 1, Len: 2, Orig: -1}, {Out: 3, Len: 1, Orig: 2}}
	if err := Write(path, "dir/a.lgo", 3, 4, spans, segs); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	for _, field := range []string{`"version":2`, `"source":"a.lgo"`, `"size":3`, `"out":4`, `"edits"`, `"segments"`} {
		if !strings.Contains(string(data), field) {
			t.Errorf("sidecar %q lacks %s", data, field)
		}
	}
	var m EditMap
	if err := json.Unmarshal(data, &m); err != nil {
		t.Fatalf("the written map does not parse: %v", err)
	}
	if m.Version != Version || m.Source != "a.lgo" || m.Size != 3 || m.Out != 4 {
		t.Errorf("map header = %+v, want version %d source a.lgo size 3 out 4", m, Version)
	}
	if !slices.Equal(m.Edits, spans) || !slices.Equal(m.Segments, segs) {
		t.Errorf("map = %+v, want spans %+v segments %+v", m, spans, segs)
	}
}

func TestCheckRejectsVersion(t *testing.T) {
	src, out := []byte("abc"), []byte("aXXc")
	edits := []EditSpan{{Start: 1, End: 2, Len: 2}}
	segs := []Segment{{Out: 0, Len: 1, Orig: 0}, {Out: 1, Len: 2, Orig: -1}, {Out: 3, Len: 1, Orig: 2}}
	for _, v := range []int{Version - 1, Version + 1, 0, -1} {
		m := goodMap("abc", "aXXc", edits, segs)
		m.Version = v
		if err := Check(src, out, m); err == nil {
			t.Errorf("version %d accepted", v)
		}
	}
	if err := Check(src, out, goodMap("abc", "aXXc", edits, segs)); err != nil {
		t.Errorf("version %d rejected: %v", Version, err)
	}
}

func TestCheckRejectsBadMaps(t *testing.T) {
	src, out := []byte("abc"), []byte("aXXc")
	edits := []EditSpan{{Start: 1, End: 2, Len: 2}}
	goodSegs := []Segment{{Out: 0, Len: 1, Orig: 0}, {Out: 1, Len: 2, Orig: -1}, {Out: 3, Len: 1, Orig: 2}}
	good := goodMap("abc", "aXXc", edits, goodSegs)
	if err := Check(src, out, good); err != nil {
		t.Fatalf("good map rejected: %v", err)
	}
	bad := func(size, outSize int, edits ...EditSpan) EditMap {
		return EditMap{Version: Version, Source: "x.lgo", Size: size, Out: outSize, Edits: edits, Segments: goodSegs}
	}
	cases := map[string]EditMap{
		"source size":  bad(4, 4, good.Edits...),
		"out size":     bad(3, 5, good.Edits...),
		"sum":          bad(3, 4, EditSpan{Start: 1, End: 2, Len: 3}),
		"past the end": bad(3, 4, EditSpan{Start: 1, End: 9, Len: 10}),
		"inverted":     bad(3, 4, EditSpan{Start: 2, End: 1, Len: 3}),
		"negative len": bad(3, 4, EditSpan{Start: 1, End: 2, Len: -1}, EditSpan{Start: 2, End: 2, Len: 2}),
		"overlap":      bad(3, 5, EditSpan{Start: 0, End: 2, Len: 2}, EditSpan{Start: 1, End: 2, Len: 2}),
		"out of order": bad(3, 4, EditSpan{Start: 2, End: 3, Len: 1}, EditSpan{Start: 0, End: 1, Len: 1}),
		"no segments": {
			Version: Version, Source: "x.lgo", Size: 3, Out: 4,
			Edits: good.Edits,
		},
		"segments short": {
			Version: Version, Source: "x.lgo", Size: 3, Out: 4,
			Edits:    good.Edits,
			Segments: []Segment{{Out: 0, Len: 3, Orig: 0}},
		},
		"segments overlap": {
			Version: Version, Source: "x.lgo", Size: 3, Out: 4,
			Edits:    good.Edits,
			Segments: []Segment{{Out: 0, Len: 2, Orig: -1}, {Out: 1, Len: 3, Orig: -1}},
		},
		"copy outside source": {
			Version: Version, Source: "x.lgo", Size: 3, Out: 4,
			Edits:    good.Edits,
			Segments: []Segment{{Out: 0, Len: 4, Orig: 3}},
		},
		"copy bytes differ": {
			Version: Version, Source: "x.lgo", Size: 3, Out: 4,
			Edits:    good.Edits,
			Segments: []Segment{{Out: 0, Len: 1, Orig: 0}, {Out: 1, Len: 2, Orig: -1}, {Out: 3, Len: 1, Orig: 0}},
		},
	}
	for name, m := range cases {
		if err := Check(src, out, m); err == nil {
			t.Errorf("%s: map %+v accepted", name, m)
		}
		if _, err := New(src, out, m); err == nil {
			t.Errorf("%s: New accepted map %+v", name, m)
		}
	}
}

func TestCheckRejectsDegenerateSegments(t *testing.T) {
	src, out := []byte("abc"), []byte("aXXc")
	edits := []EditSpan{{Start: 1, End: 2, Len: 2}}
	cases := map[string][]Segment{
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
	for name, segs := range cases {
		m := goodMap("abc", "aXXc", edits, segs)
		if err := Check(src, out, m); err == nil {
			t.Errorf("%s: degenerate map accepted", name)
		}
	}
}

func TestEmptyMap(t *testing.T) {
	pm := mustNew(t, "", "", goodMap("", "", nil, nil))
	if got, ok := pm.LoweredOffset(0, Trailing); !ok || got != 0 {
		t.Errorf("trailing offset zero maps to (%d, %v), want (0, true)", got, ok)
	}
	if got, ok := pm.LoweredOffset(0, Leading); !ok || got != 0 {
		t.Errorf("leading offset zero maps to (%d, %v), want (0, true)", got, ok)
	}
	if got, gen := pm.OriginalOffset(0, Leading); gen || got != 0 {
		t.Errorf("lowered offset zero maps to (%d, %v), want (0, false)", got, gen)
	}
	if got := pm.LoweredOffsetNear(0, Leading); got != 0 {
		t.Errorf("near mapping gave %d, want 0", got)
	}
}

func TestIdentitySegmentsRoundTrip(t *testing.T) {
	const src = "abcdefghij"
	pm := mustNew(t, src, src, goodMap(src, src, nil, []Segment{{Out: 0, Len: len(src), Orig: 0}}))
	for o := 0; o <= len(src); o++ {
		for _, e := range []Edge{Leading, Trailing} {
			got, ok := pm.LoweredOffset(o, e)
			if !ok || got != o {
				t.Errorf("source %d edge %d maps to (%d, %v), want (%d, true)", o, e, got, ok, o)
			}
			back, gen := pm.OriginalOffset(o, e)
			if gen || back != o {
				t.Errorf("lowered %d edge %d maps to (%d, %v), want (%d, false)", o, e, back, gen, o)
			}
		}
	}
}

func TestSegmentEdgeNeighbors(t *testing.T) {
	const src = "abcdefghij"
	const out = "abcdeXYfgij"
	edits := []EditSpan{{Start: 5, End: 5, Len: 2}, {Start: 7, End: 8, Len: 0}}
	segs := []Segment{
		{Out: 0, Len: 5, Orig: 0},
		{Out: 5, Len: 2, Orig: -1},
		{Out: 7, Len: 2, Orig: 5},
		{Out: 9, Len: 2, Orig: 8},
	}
	pm := mustNew(t, src, out, goodMap(src, out, edits, segs))

	lowered := []struct {
		o       int
		e       Edge
		want    int
		wantOK  bool
		comment string
	}{
		{4, Trailing, 4, true, "last byte before the insertion"},
		{5, Trailing, 5, true, "insertion point stays before XY"},
		{5, Leading, 7, true, "insertion point goes after XY"},
		{6, Trailing, 8, true, "first copied byte after the insertion"},
		{7, Trailing, 9, true, "deleted byte trailing edge sits after fg"},
		{7, Leading, 9, true, "deleted byte leading edge sits after fg"},
		{8, Trailing, 9, true, "first byte after the deletion"},
		{9, Leading, 10, true, "last source byte"},
		{10, Trailing, 11, true, "end of source before nothing appended"},
		{10, Leading, 11, true, "end of source"},
	}
	for _, c := range lowered {
		got, ok := pm.LoweredOffset(c.o, c.e)
		if !ok || got != c.want {
			t.Errorf("LoweredOffset(%d, %d) = (%d, %v), want (%d, %v): %s", c.o, c.e, got, ok, c.want, c.wantOK, c.comment)
		}
	}

	original := []struct {
		o       int
		e       Edge
		want    int
		wantGen bool
		comment string
	}{
		{4, Leading, 4, false, "inside the first copied run"},
		{5, Leading, 5, false, "generated run start counts as user text"},
		{5, Trailing, 5, false, "generated run start trails to the same boundary"},
		{6, Leading, 5, true, "inside generated XY anchors to the insertion point"},
		{7, Leading, 5, false, "generated run end counts as user text"},
		{8, Leading, 6, false, "copied g maps exactly"},
		{9, Leading, 8, false, "generated run start before the deleted byte"},
		{9, Trailing, 7, false, "deletion seam trailing edge stays before the deleted byte"},
		{10, Leading, 9, false, "last lowered byte"},
		{11, Leading, 10, false, "end of lowered text"},
		{11, Trailing, 10, false, "end of lowered text trailing edge"},
	}
	for _, c := range original {
		got, gen := pm.OriginalOffset(c.o, c.e)
		if gen != c.wantGen || got != c.want {
			t.Errorf("OriginalOffset(%d, %d) = (%d, %v), want (%d, %v): %s", c.o, c.e, got, gen, c.want, c.wantGen, c.comment)
		}
	}
}

func TestDeletedByteNearMapping(t *testing.T) {
	const src = "one\ndropped\ntwo\n"
	const out = "one\ntwo\n"
	edits := []EditSpan{{Start: 4, End: 12, Len: 0}}
	segs := []Segment{{Out: 0, Len: 4, Orig: 0}, {Out: 4, Len: 4, Orig: 12}}
	pm := mustNew(t, src, out, goodMap(src, out, edits, segs))
	for o := 4; o < 12; o++ {
		if got := pm.LoweredOffsetNear(o, Leading); got != 4 {
			t.Errorf("LoweredOffsetNear(%d, leading) = %d, want 4", o, got)
		}
		if got := pm.LoweredOffsetNear(o, Trailing); got != 4 {
			t.Errorf("LoweredOffsetNear(%d, trailing) = %d, want 4", o, got)
		}
	}
	if got := pm.LoweredOffsetNear(3, Leading); got != 3 {
		t.Errorf("LoweredOffsetNear(3, leading) = %d, want 3 inside the first run", got)
	}
}

func TestMovedAndDuplicatedChunks(t *testing.T) {
	src := []byte("ABCDE FGHIJ")
	movedOut := []byte("FGHIJ ABCDE!")
	moved := goodMap(string(src), string(movedOut),
		[]EditSpan{{Start: 0, End: len(src), Len: len(movedOut)}},
		[]Segment{
			{Out: 0, Len: 5, Orig: 6},
			{Out: 5, Len: 1, Orig: -1},
			{Out: 6, Len: 5, Orig: 0},
			{Out: 11, Len: 1, Orig: -1},
		})
	pm, err := New(src, movedOut, moved)
	if err != nil {
		t.Fatal(err)
	}
	checkExact(t, src, movedOut, pm)

	dupOut := []byte("AB AB")
	dup := goodMap("AB", string(dupOut),
		[]EditSpan{{Start: 0, End: 0, Len: 1}, {Start: 2, End: 2, Len: 2}},
		[]Segment{
			{Out: 0, Len: 2, Orig: 0},
			{Out: 2, Len: 1, Orig: -1},
			{Out: 3, Len: 2, Orig: 0},
		})
	dpm, err := New([]byte("AB"), dupOut, dup)
	if err != nil {
		t.Fatal(err)
	}
	checkExact(t, []byte("AB"), dupOut, dpm)
}

func TestLargeReplacementMapsBySegments(t *testing.T) {
	src := []byte("w" + strings.Repeat("(", 2048))
	mid := src[512:1536]
	out := append(append([]byte("gen<"), mid...), []byte(">gen")...)
	m := goodMap(string(src), string(out),
		[]EditSpan{{Start: 0, End: len(src), Len: len(out)}},
		[]Segment{
			{Out: 0, Len: 4, Orig: -1},
			{Out: 4, Len: len(mid), Orig: 512},
			{Out: 4 + len(mid), Len: 4, Orig: -1},
		})
	pm, err := New(src, out, m)
	if err != nil {
		t.Fatal(err)
	}
	for i := range mid {
		got, ok := pm.LoweredOffset(512+i, Leading)
		if !ok || got != 4+i {
			t.Fatalf("copied original %d maps to (%d, %v), want (%d, true)", 512+i, got, ok, 4+i)
		}
		back, gen := pm.OriginalOffset(4+i, Leading)
		if gen || back != 512+i {
			t.Fatalf("copied lowered %d maps to (%d, %v), want (%d, false)", 4+i, back, gen, 512+i)
		}
	}
	if got, ok := pm.LoweredOffset(0, Leading); ok || got != 0 {
		t.Errorf("deleted original 0 maps to (%d, %v), want (0, false)", got, ok)
	}
	if got, gen := pm.OriginalOffset(1, Leading); !gen || got != 0 {
		t.Errorf("generated prefix maps to (%d, %v), want (0, true)", got, gen)
	}
	if got, gen := pm.OriginalOffset(len(out)-2, Leading); !gen || got != 1536 {
		t.Errorf("generated suffix maps to (%d, %v), want (1536, true)", got, gen)
	}
}

func checkExact(t *testing.T, src, out []byte, pm *Map) {
	t.Helper()
	for _, s := range pm.Segments {
		if s.Orig < 0 {
			continue
		}
		for i := range s.Len {
			if out[s.Out+i] != src[s.Orig+i] {
				t.Fatalf("copied lowered %d (%q) differs from original %d (%q)", s.Out+i, out[s.Out+i], s.Orig+i, src[s.Orig+i])
			}
			if back, gen := pm.OriginalOffset(s.Out+i, Leading); gen || back != s.Orig+i {
				t.Fatalf("lowered %d maps to (%d, %v), want (%d, false)", s.Out+i, back, gen, s.Orig+i)
			}
			if low, ok := pm.LoweredOffset(s.Orig+i, Leading); !ok {
				t.Fatalf("original %d has no lowered position", s.Orig+i)
			} else if back, gen := pm.OriginalOffset(low, Leading); gen || back != s.Orig+i {
				t.Fatalf("original %d -> %d -> (%d, %v) is not the identity", s.Orig+i, low, back, gen)
			}
		}
		if start, _ := pm.OriginalOffset(s.Out, Leading); start != s.Orig {
			t.Fatalf("run start at %d maps to %d, want %d", s.Out, start, s.Orig)
		}
		if finish, _ := pm.OriginalOffset(s.Out+s.Len, Trailing); finish != s.Orig+s.Len {
			t.Fatalf("run end at %d maps to %d, want %d", s.Out+s.Len, finish, s.Orig+s.Len)
		}
	}
}

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package slog

import (
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"
)

func TestRecordAttrs(t *testing.T) {
	as := []Attr{Int("k1", 1), String("k2", "foo"), Int("k3", 3),
		Int64("k4", -1), Float64("f", 3.1), Uint64("u", 999)}
	r := newRecordWithAttrs(as)
	if g, w := r.NumAttrs(), len(as); g != w {
		t.Errorf("NumAttrs: got %d, want %d", g, w)
	}
	if got := attrsSlice(r); !attrsEqual(got, as) {
		t.Errorf("got %v, want %v", got, as)
	}

	// Early return.
	// Hit both loops in Record.Attrs: front and back.
	for _, stop := range []int{2, 6} {
		var got []Attr
		r.Attrs(func { a ->
			got = append(got, a)
			return len(got) < stop
		})
		want := as[:stop]
		if !attrsEqual(got, want) {
			t.Errorf("got %v, want %v", got, want)
		}
	}
}

func TestRecordSource(t *testing.T) {
	// Zero call depth => empty *Source.
	for _, test := range []struct {
		depth            int
		wantFunction     string
		wantFile         string
		wantLinePositive bool
	}{
		{0, "", "", false},
		{-16, "", "", false},
		{1, "log/slog.TestRecordSource", "record_test.go", true}, // 1: caller of NewRecord
		{2, "testing.tRunner", "testing.go", true},
	} {
		var pc uintptr
		if test.depth > 0 {
			pc = callerPC(test.depth + 1)
		}
		r := NewRecord(time.Time{}, 0, "", pc)
		got := r.source()
		if i := strings.LastIndexByte(got.File, '/'); i >= 0 {
			got.File = got.File[i+1:]
		}
		if got.Function != test.wantFunction || got.File != test.wantFile || (got.Line > 0) != test.wantLinePositive {
			t.Errorf("depth %d: got (%q, %q, %d), want (%q, %q, %t)",
				test.depth,
				got.Function, got.File, got.Line,
				test.wantFunction, test.wantFile, test.wantLinePositive)
		}
	}
}

func TestAliasingAndClone(t *testing.T) {
	intAttrs := func(from, to int) []Attr {
		var as []Attr
		for i := from; i < to; i++ {
			as = append(as, Int("k", i))
		}
		return as
	}

	check := func(r Record, want []Attr) {
		t.Helper()
		got := attrsSlice(r)
		if !attrsEqual(got, want) {
			t.Errorf("got %v, want %v", got, want)
		}
	}

	// Create a record whose Attrs overflow the inline array,
	// creating a slice in r.back.
	r1 := NewRecord(time.Time{}, 0, "", 0)
	r1.AddAttrs(intAttrs(0, nAttrsInline+1)...)
	// Ensure that r1.back's capacity exceeds its length.
	b := make([]Attr, len(r1.back), len(r1.back)+1)
	copy(b, r1.back)
	r1.back = b
	// Make a copy that shares state.
	r2 := r1
	// Adding to both should insert a special Attr in the second.
	r1AttrsBefore := attrsSlice(r1)
	r1.AddAttrs(Int("p", 0))
	r2.AddAttrs(Int("p", 1))
	check(r1, append(slices.Clip(r1AttrsBefore), Int("p", 0)))
	r1Attrs := attrsSlice(r1)
	check(r2, append(slices.Clip(r1AttrsBefore),
		String("!BUG", "AddAttrs unsafely called on copy of Record made without using Record.Clone"), Int("p", 1)))

	// Adding to a clone is fine.
	r2 = r1.Clone()
	check(r2, r1Attrs)
	r2.AddAttrs(Int("p", 2))
	check(r1, r1Attrs) // r1 is unchanged
	check(r2, append(slices.Clip(r1Attrs), Int("p", 2)))
}

func newRecordWithAttrs(as []Attr) Record {
	r := NewRecord(time.Now(), LevelInfo, "", 0)
	r.AddAttrs(as...)
	return r
}

func attrsSlice(r Record) []Attr {
	s := make([]Attr, 0, r.NumAttrs())
	r.Attrs(func { a ->
		s = append(s, a)
		return true
	})
	return s
}

func attrsEqual(as1, as2 []Attr) bool {
	return slices.EqualFunc(as1, as2, Attr.Equal)
}

// Currently, pc(2) takes over 400ns, which is too expensive
// to call it for every log message.
func BenchmarkPC(b *testing.B) {
	for depth := 0; depth < 5; depth++ {
		b.Run(strconv.Itoa(depth), func { b ->
			b.ReportAllocs()
			var x uintptr
			for i := 0; i < b.N; i++ {
				x = callerPC(depth)
			}
			_ = x
		})
	}
}

func BenchmarkRecord(b *testing.B) {
	const nAttrs = nAttrsInline * 10
	var a Attr

	for i := 0; i < b.N; i++ {
		r := NewRecord(time.Time{}, LevelInfo, "", 0)
		for j := 0; j < nAttrs; j++ {
			r.AddAttrs(Int("k", j))
		}
		r.Attrs(func { b ->
			a = b
			return true
		})
	}
	_ = a
}

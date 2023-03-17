// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package slog

import (
	"log/slog/internal/buffer"
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
}

func TestRecordSourceLine(t *testing.T) {
	// Zero call depth => empty file/line
	for _, test := range []struct {
		depth            int
		wantFile         string
		wantLinePositive bool
	}{
		{0, "", false},
		{-16, "", false},
		{1, "record_test.go", true}, // 1: caller of NewRecord
		{2, "testing.go", true},
	} {
		var pc uintptr
		if test.depth > 0 {
			pc = callerPC(test.depth + 1)
		}
		r := NewRecord(time.Time{}, 0, "", pc)
		gotFile, gotLine := sourceLine(r)
		if i := strings.LastIndexByte(gotFile, '/'); i >= 0 {
			gotFile = gotFile[i+1:]
		}
		if gotFile != test.wantFile || (gotLine > 0) != test.wantLinePositive {
			t.Errorf("depth %d: got (%q, %d), want (%q, %t)",
				test.depth, gotFile, gotLine, test.wantFile, test.wantLinePositive)
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
	// Adding to both should panic.
	r1.AddAttrs(Int("p", 0))
	if !panics(func() { r2.AddAttrs(Int("p", 1)) }) {
		t.Error("expected panic")
	}
	r1Attrs := attrsSlice(r1)
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
	r.Attrs(func(a Attr) { s = append(s, a) })
	return s
}

func attrsEqual(as1, as2 []Attr) bool {
	return slices.EqualFunc(as1, as2, Attr.Equal)
}

// Currently, pc(2) takes over 400ns, which is too expensive
// to call it for every log message.
func BenchmarkPC(b *testing.B) {
	for depth := 0; depth < 5; depth++ {
		b.Run(strconv.Itoa(depth), func(b *testing.B) {
			b.ReportAllocs()
			var x uintptr
			for i := 0; i < b.N; i++ {
				x = callerPC(depth)
			}
			_ = x
		})
	}
}

func BenchmarkSourceLine(b *testing.B) {
	r := NewRecord(time.Now(), LevelInfo, "", 5)
	b.Run("alone", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			file, line := sourceLine(r)
			_ = file
			_ = line
		}
	})
	b.Run("stringifying", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			file, line := sourceLine(r)
			buf := buffer.New()
			buf.WriteString(file)
			buf.WriteByte(':')
			buf.WritePosInt(line)
			s := buf.String()
			buf.Free()
			_ = s
		}
	})
}

func BenchmarkRecord(b *testing.B) {
	const nAttrs = nAttrsInline * 10
	var a Attr

	for i := 0; i < b.N; i++ {
		r := NewRecord(time.Time{}, LevelInfo, "", 0)
		for j := 0; j < nAttrs; j++ {
			r.AddAttrs(Int("k", j))
		}
		r.Attrs(func(b Attr) { a = b })
	}
	_ = a
}

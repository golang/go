// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package slog

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"internal/testenv"
	"io"
	"strings"
	"testing"
	"time"
)

var testTime = time.Date(2000, 1, 2, 3, 4, 5, 0, time.UTC)

func TestTextHandler(t *testing.T) {
	for _, test := range []struct {
		name             string
		attr             Attr
		wantKey, wantVal string
	}{
		{
			"unquoted",
			Int("a", 1),
			"a", "1",
		},
		{
			"quoted",
			String("x = y", `qu"o`),
			`"x = y"`, `"qu\"o"`,
		},
		{
			"String method",
			Any("name", name{"Ren", "Hoek"}),
			`name`, `"Hoek, Ren"`,
		},
		{
			"struct",
			Any("x", &struct{ A, b int }{A: 1, b: 2}),
			`x`, `"&{A:1 b:2}"`,
		},
		{
			"TextMarshaler",
			Any("t", text{"abc"}),
			`t`, `"text{\"abc\"}"`,
		},
		{
			"TextMarshaler error",
			Any("t", text{""}),
			`t`, `"!ERROR:text: empty string"`,
		},
		{
			"nil value",
			Any("a", nil),
			`a`, `<nil>`,
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			for _, opts := range []struct {
				name       string
				opts       HandlerOptions
				wantPrefix string
				modKey     func(string) string
			}{
				{
					"none",
					HandlerOptions{},
					`time=2000-01-02T03:04:05.000Z level=INFO msg="a message"`,
					func(s string) string { return s },
				},
				{
					"replace",
					HandlerOptions{ReplaceAttr: upperCaseKey},
					`TIME=2000-01-02T03:04:05.000Z LEVEL=INFO MSG="a message"`,
					strings.ToUpper,
				},
			} {
				t.Run(opts.name, func(t *testing.T) {
					var buf bytes.Buffer
					h := NewTextHandler(&buf, &opts.opts)
					r := NewRecord(testTime, LevelInfo, "a message", 0)
					r.AddAttrs(test.attr)
					if err := h.Handle(context.Background(), r); err != nil {
						t.Fatal(err)
					}
					got := buf.String()
					// Remove final newline.
					got = got[:len(got)-1]
					want := opts.wantPrefix + " " + opts.modKey(test.wantKey) + "=" + test.wantVal
					if got != want {
						t.Errorf("\ngot  %s\nwant %s", got, want)
					}
				})
			}
		})
	}
}

// for testing fmt.Sprint
type name struct {
	First, Last string
}

func (n name) String() string { return n.Last + ", " + n.First }

// for testing TextMarshaler
type text struct {
	s string
}

func (t text) String() string { return t.s } // should be ignored

func (t text) MarshalText() ([]byte, error) {
	if t.s == "" {
		return nil, errors.New("text: empty string")
	}
	return []byte(fmt.Sprintf("text{%q}", t.s)), nil
}

func TestTextHandlerPreformatted(t *testing.T) {
	var buf bytes.Buffer
	var h Handler = NewTextHandler(&buf, nil)
	h = h.WithAttrs([]Attr{Duration("dur", time.Minute), Bool("b", true)})
	// Also test omitting time.
	r := NewRecord(time.Time{}, 0 /* 0 Level is INFO */, "m", 0)
	r.AddAttrs(Int("a", 1))
	if err := h.Handle(context.Background(), r); err != nil {
		t.Fatal(err)
	}
	got := strings.TrimSuffix(buf.String(), "\n")
	want := `level=INFO msg=m dur=1m0s b=true a=1`
	if got != want {
		t.Errorf("got %s, want %s", got, want)
	}
}

func TestTextHandlerAlloc(t *testing.T) {
	testenv.SkipIfOptimizationOff(t)
	r := NewRecord(time.Now(), LevelInfo, "msg", 0)
	for i := 0; i < 10; i++ {
		r.AddAttrs(Int("x = y", i))
	}
	var h Handler = NewTextHandler(io.Discard, nil)
	wantAllocs(t, 0, func() { h.Handle(context.Background(), r) })

	h = h.WithGroup("s")
	r.AddAttrs(Group("g", Int("a", 1)))
	wantAllocs(t, 0, func() { h.Handle(context.Background(), r) })
}

func TestNeedsQuoting(t *testing.T) {
	for _, test := range []struct {
		in   string
		want bool
	}{
		{"", true},
		{"ab", false},
		{"a=b", true},
		{`"ab"`, true},
		{"\a\b", true},
		{"a\tb", true},
		{"µåπ", false},
		{"a b", true},
		{"badutf8\xF6", true},
	} {
		got := needsQuotingString(test.in)
		if got != test.want {
			t.Errorf("needsQuotingString: %q: got %t, want %t", test.in, got, test.want)
		}

		got = needsQuotingBytes([]byte(test.in))
		if got != test.want {
			t.Errorf("needsQuotingBytes: %q: got %t, want %t", test.in, got, test.want)
		}
	}
}

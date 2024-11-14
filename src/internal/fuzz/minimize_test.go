// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin || freebsd || linux || windows

package fuzz

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"reflect"
	"testing"
	"time"
	"unicode"
	"unicode/utf8"
)

func TestMinimizeInput(t *testing.T) {
	type testcase struct {
		name     string
		fn       func(CorpusEntry) error
		input    []any
		expected []any
	}
	cases := []testcase{
		{
			name: "ones_byte",
			fn: func(e CorpusEntry) error {
				b := e.Values[0].([]byte)
				ones := 0
				for _, v := range b {
					if v == 1 {
						ones++
					}
				}
				if ones == 3 {
					return fmt.Errorf("bad %v", e.Values[0])
				}
				return nil
			},
			input:    []any{[]byte{0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
			expected: []any{[]byte{1, 1, 1}},
		},
		{
			name: "single_bytes",
			fn: func(e CorpusEntry) error {
				b := e.Values[0].([]byte)
				if len(b) < 2 {
					return nil
				}
				if len(b) == 2 && b[0] == 1 && b[1] == 2 {
					return nil
				}
				return fmt.Errorf("bad %v", e.Values[0])
			},
			input:    []any{[]byte{1, 2, 3, 4, 5}},
			expected: []any{[]byte("00")},
		},
		{
			name: "set_of_bytes",
			fn: func(e CorpusEntry) error {
				b := e.Values[0].([]byte)
				if len(b) < 3 {
					return nil
				}
				if bytes.Equal(b, []byte{0, 1, 2, 3, 4, 5}) || bytes.Equal(b, []byte{0, 4, 5}) {
					return fmt.Errorf("bad %v", e.Values[0])
				}
				return nil
			},
			input:    []any{[]byte{0, 1, 2, 3, 4, 5}},
			expected: []any{[]byte{0, 4, 5}},
		},
		{
			name: "non_ascii_bytes",
			fn: func(e CorpusEntry) error {
				b := e.Values[0].([]byte)
				if len(b) == 3 {
					return fmt.Errorf("bad %v", e.Values[0])
				}
				return nil
			},
			input:    []any{[]byte("ท")}, // ท is 3 bytes
			expected: []any{[]byte("000")},
		},
		{
			name: "ones_string",
			fn: func(e CorpusEntry) error {
				b := e.Values[0].(string)
				ones := 0
				for _, v := range b {
					if v == '1' {
						ones++
					}
				}
				if ones == 3 {
					return fmt.Errorf("bad %v", e.Values[0])
				}
				return nil
			},
			input:    []any{"001010001000000000000000000"},
			expected: []any{"111"},
		},
		{
			name: "string_length",
			fn: func(e CorpusEntry) error {
				b := e.Values[0].(string)
				if len(b) == 5 {
					return fmt.Errorf("bad %v", e.Values[0])
				}
				return nil
			},
			input:    []any{"zzzzz"},
			expected: []any{"00000"},
		},
		{
			name: "string_with_letter",
			fn: func(e CorpusEntry) error {
				b := e.Values[0].(string)
				r, _ := utf8.DecodeRune([]byte(b))
				if unicode.IsLetter(r) {
					return fmt.Errorf("bad %v", e.Values[0])
				}
				return nil
			},
			input:    []any{"ZZZZZ"},
			expected: []any{"A"},
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func { t ->
			t.Parallel()
			ws := &workerServer{
				fuzzFn: func(e CorpusEntry) (time.Duration, error) {
					return time.Second, tc.fn(e)
				},
			}
			mem := &sharedMem{region: make([]byte, 100)} // big enough to hold value and header
			vals := tc.input
			success, err := ws.minimizeInput(context.Background(), vals, mem, minimizeArgs{})
			if !success {
				t.Errorf("minimizeInput did not succeed")
			}
			if err == nil {
				t.Fatal("minimizeInput didn't provide an error")
			}
			if expected := fmt.Sprintf("bad %v", tc.expected[0]); err.Error() != expected {
				t.Errorf("unexpected error: got %q, want %q", err, expected)
			}
			if !reflect.DeepEqual(vals, tc.expected) {
				t.Errorf("unexpected results: got %v, want %v", vals, tc.expected)
			}
		})
	}
}

// TestMinimizeFlaky checks that if we're minimizing an interesting
// input and a flaky failure occurs, that minimization was not indicated
// to be successful, and the error isn't returned (since it's flaky).
func TestMinimizeFlaky(t *testing.T) {
	ws := &workerServer{fuzzFn: func(e CorpusEntry) (time.Duration, error) {
		return time.Second, errors.New("ohno")
	}}
	mem := &sharedMem{region: make([]byte, 100)} // big enough to hold value and header
	vals := []any{[]byte(nil)}
	args := minimizeArgs{KeepCoverage: make([]byte, len(coverageSnapshot))}
	success, err := ws.minimizeInput(context.Background(), vals, mem, args)
	if success {
		t.Error("unexpected success")
	}
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if count := mem.header().count; count != 1 {
		t.Errorf("count: got %d, want 1", count)
	}
}

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin || linux || windows

package fuzz

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"reflect"
	"testing"
	"unicode"
	"unicode/utf8"
)

func TestMinimizeInput(t *testing.T) {
	type testcase struct {
		name     string
		fn       func(CorpusEntry) error
		input    []interface{}
		expected []interface{}
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
			input:    []interface{}{[]byte{0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
			expected: []interface{}{[]byte{1, 1, 1}},
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
			input:    []interface{}{[]byte{1, 2, 3, 4, 5}},
			expected: []interface{}{[]byte("00")},
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
			input:    []interface{}{[]byte{0, 1, 2, 3, 4, 5}},
			expected: []interface{}{[]byte{0, 4, 5}},
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
			input:    []interface{}{[]byte("ท")}, // ท is 3 bytes
			expected: []interface{}{[]byte("000")},
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
			input:    []interface{}{"001010001000000000000000000"},
			expected: []interface{}{"111"},
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
			input:    []interface{}{"zzzzz"},
			expected: []interface{}{"00000"},
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
			input:    []interface{}{"ZZZZZ"},
			expected: []interface{}{"A"},
		},
		{
			name: "int",
			fn: func(e CorpusEntry) error {
				i := e.Values[0].(int)
				if i > 100 {
					return fmt.Errorf("bad %v", e.Values[0])
				}
				return nil
			},
			input:    []interface{}{123456},
			expected: []interface{}{123},
		},
		{
			name: "int8",
			fn: func(e CorpusEntry) error {
				i := e.Values[0].(int8)
				if i > 10 {
					return fmt.Errorf("bad %v", e.Values[0])
				}
				return nil
			},
			input:    []interface{}{int8(1<<7 - 1)},
			expected: []interface{}{int8(12)},
		},
		{
			name: "int16",
			fn: func(e CorpusEntry) error {
				i := e.Values[0].(int16)
				if i > 10 {
					return fmt.Errorf("bad %v", e.Values[0])
				}
				return nil
			},
			input:    []interface{}{int16(1<<15 - 1)},
			expected: []interface{}{int16(32)},
		},
		{
			fn: func(e CorpusEntry) error {
				i := e.Values[0].(int32)
				if i > 10 {
					return fmt.Errorf("bad %v", e.Values[0])
				}
				return nil
			},
			input:    []interface{}{int32(1<<31 - 1)},
			expected: []interface{}{int32(21)},
		},
		{
			name: "int32",
			fn: func(e CorpusEntry) error {
				i := e.Values[0].(uint)
				if i > 10 {
					return fmt.Errorf("bad %v", e.Values[0])
				}
				return nil
			},
			input:    []interface{}{uint(123456)},
			expected: []interface{}{uint(12)},
		},
		{
			name: "uint8",
			fn: func(e CorpusEntry) error {
				i := e.Values[0].(uint8)
				if i > 10 {
					return fmt.Errorf("bad %v", e.Values[0])
				}
				return nil
			},
			input:    []interface{}{uint8(1<<8 - 1)},
			expected: []interface{}{uint8(25)},
		},
		{
			name: "uint16",
			fn: func(e CorpusEntry) error {
				i := e.Values[0].(uint16)
				if i > 10 {
					return fmt.Errorf("bad %v", e.Values[0])
				}
				return nil
			},
			input:    []interface{}{uint16(1<<16 - 1)},
			expected: []interface{}{uint16(65)},
		},
		{
			name: "uint32",
			fn: func(e CorpusEntry) error {
				i := e.Values[0].(uint32)
				if i > 10 {
					return fmt.Errorf("bad %v", e.Values[0])
				}
				return nil
			},
			input:    []interface{}{uint32(1<<32 - 1)},
			expected: []interface{}{uint32(42)},
		},
		{
			name: "float32",
			fn: func(e CorpusEntry) error {
				if i := e.Values[0].(float32); i == 1.23 {
					return nil
				}
				return fmt.Errorf("bad %v", e.Values[0])
			},
			input:    []interface{}{float32(1.23456789)},
			expected: []interface{}{float32(1.2)},
		},
		{
			name: "float64",
			fn: func(e CorpusEntry) error {
				if i := e.Values[0].(float64); i == 1.23 {
					return nil
				}
				return fmt.Errorf("bad %v", e.Values[0])
			},
			input:    []interface{}{float64(1.23456789)},
			expected: []interface{}{float64(1.2)},
		},
	}

	// If we are on a 64 bit platform add int64 and uint64 tests
	if v := int64(1<<63 - 1); int64(int(v)) == v {
		cases = append(cases, testcase{
			name: "int64",
			fn: func(e CorpusEntry) error {
				i := e.Values[0].(int64)
				if i > 10 {
					return fmt.Errorf("bad %v", e.Values[0])
				}
				return nil
			},
			input:    []interface{}{int64(1<<63 - 1)},
			expected: []interface{}{int64(92)},
		}, testcase{
			name: "uint64",
			fn: func(e CorpusEntry) error {
				i := e.Values[0].(uint64)
				if i > 10 {
					return fmt.Errorf("bad %v", e.Values[0])
				}
				return nil
			},
			input:    []interface{}{uint64(1<<64 - 1)},
			expected: []interface{}{uint64(18)},
		})
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			ws := &workerServer{
				fuzzFn: tc.fn,
			}
			count := int64(0)
			vals := tc.input
			success, err := ws.minimizeInput(context.Background(), vals, &count, 0, nil)
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
	ws := &workerServer{fuzzFn: func(e CorpusEntry) error {
		return errors.New("ohno")
	}}
	keepCoverage := make([]byte, len(coverageSnapshot))
	count := int64(0)
	vals := []interface{}{[]byte(nil)}
	success, err := ws.minimizeInput(context.Background(), vals, &count, 0, keepCoverage)
	if success {
		t.Error("unexpected success")
	}
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if count != 1 {
		t.Errorf("count: got %d, want 1", count)
	}
}

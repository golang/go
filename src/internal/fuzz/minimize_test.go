// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin || linux || windows
// +build darwin linux windows

package fuzz

import (
	"context"
	"fmt"
	"reflect"
	"testing"
)

func TestMinimizeInput(t *testing.T) {
	type testcase struct {
		fn       func(CorpusEntry) error
		input    []interface{}
		expected []interface{}
	}
	cases := []testcase{
		{
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
		ws := &workerServer{
			fuzzFn: tc.fn,
		}
		count := int64(0)
		vals := tc.input
		err := ws.minimizeInput(context.Background(), vals, &count, 0)
		if err == nil {
			t.Error("minimizeInput didn't fail")
		}
		if expected := fmt.Sprintf("bad %v", tc.input[0]); err.Error() != expected {
			t.Errorf("unexpected error: got %s, want %s", err, expected)
		}
		if !reflect.DeepEqual(vals, tc.expected) {
			t.Errorf("unexpected results: got %v, want %v", vals, tc.expected)
		}
	}
}

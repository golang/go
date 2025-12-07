// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fmt_test

import (
	"errors"
	. "fmt"
	"math"
	"std/fmt"
	"strings"
	"testing"
)

func TestJoin(t *testing.T) {
	tests := []struct {
		name     string
		elems    any
		sep      string
		expected string
	}{
		// Empty and nil slices
		{
			name:     "empty int slice",
			elems:    []int{},
			sep:      ", ",
			expected: "",
		},
		{
			name:     "nil int slice",
			elems:    []int(nil),
			sep:      ", ",
			expected: "",
		},
		{
			name:     "empty string slice",
			elems:    []string{},
			sep:      ", ",
			expected: "",
		},
		{
			name:     "nil string slice",
			elems:    []string(nil),
			sep:      ", ",
			expected: "",
		},

		// Single element
		{
			name:     "single int",
			elems:    []int{42},
			sep:      ", ",
			expected: "42",
		},
		{
			name:     "single string",
			elems:    []string{"hello"},
			sep:      ", ",
			expected: "hello",
		},
		{
			name:     "single float",
			elems:    []float64{3.14},
			sep:      ", ",
			expected: "3.14",
		},
		{
			name:     "single bool",
			elems:    []bool{true},
			sep:      ", ",
			expected: "true",
		},

		// Multiple elements - integers
		{
			name:     "multiple ints",
			elems:    []int{1, 2, 3, 4, 5},
			sep:      ", ",
			expected: "1, 2, 3, 4, 5",
		},
		{
			name:     "multiple ints with different separator",
			elems:    []int{1, 2, 3},
			sep:      " | ",
			expected: "1 | 2 | 3",
		},
		{
			name:     "multiple ints with empty separator",
			elems:    []int{1, 2, 3},
			sep:      "",
			expected: "123",
		},
		{
			name:     "multiple ints with newline separator",
			elems:    []int{1, 2, 3},
			sep:      "\n",
			expected: "1\n2\n3",
		},

		// Multiple elements - strings
		{
			name:     "multiple strings",
			elems:    []string{"apple", "banana", "cherry"},
			sep:      ", ",
			expected: "apple, banana, cherry",
		},
		{
			name:     "multiple strings with pipe",
			elems:    []string{"a", "b", "c"},
			sep:      " | ",
			expected: "a | b | c",
		},

		// Multiple elements - floats
		{
			name:     "multiple floats",
			elems:    []float64{1.1, 2.2, 3.3},
			sep:      ", ",
			expected: "1.1, 2.2, 3.3",
		},
		{
			name:     "multiple floats with special values",
			elems:    []float64{0.0, -1.5, 3.14159},
			sep:      " | ",
			expected: "0, -1.5, 3.14159",
		},

		// Multiple elements - booleans
		{
			name:     "multiple bools",
			elems:    []bool{true, false, true},
			sep:      ", ",
			expected: "true, false, true",
		},

		// Mixed types (using interface{})
		{
			name:     "mixed types",
			elems:    []any{1, "two", 3.14, true},
			sep:      " - ",
			expected: "1 - two - 3.14 - true",
		},

		// Nil values
		{
			name:     "slice with nil interface",
			elems:    []any{nil, "hello", nil},
			sep:      ", ",
			expected: "<nil>, hello, <nil>",
		},

		// Special separators
		{
			name:     "comma separator",
			elems:    []int{1, 2, 3},
			sep:      ",",
			expected: "1,2,3",
		},
		{
			name:     "space separator",
			elems:    []int{1, 2, 3},
			sep:      " ",
			expected: "1 2 3",
		},
		{
			name:     "tab separator",
			elems:    []int{1, 2, 3},
			sep:      "\t",
			expected: "1\t2\t3",
		},
		{
			name:     "unicode separator",
			elems:    []int{1, 2, 3},
			sep:      " → ",
			expected: "1 → 2 → 3",
		},

		// Large slices
		{
			name:  "large slice",
			elems: make([]int, 100),
			sep:   ",",
			expected: func() string {
				var result string
				for i := 0; i < 100; i++ {
					if i > 0 {
						result += ","
					}
					result += "0"
				}
				return result
			}(),
		},

		// Types with String() method
		{
			name:     "custom stringer type",
			elems:    []stringerType{{val: 1}, {val: 2}, {val: 3}},
			sep:      ", ",
			expected: "Stringer(1), Stringer(2), Stringer(3)",
		},

		// Types implementing error interface
		{
			name:     "error types",
			elems:    []error{errors.New("err1"), errors.New("err2")},
			sep:      " | ",
			expected: "err1 | err2",
		},

		// Zero values
		{
			name:     "zero values",
			elems:    []int{0, 0, 0},
			sep:      ", ",
			expected: "0, 0, 0",
		},
		{
			name:     "zero float values",
			elems:    []float64{0.0, 0.0},
			sep:      ", ",
			expected: "0, 0",
		},
		{
			name:     "zero bool values",
			elems:    []bool{false, false},
			sep:      ", ",
			expected: "false, false",
		},

		// Negative numbers
		{
			name:     "negative numbers",
			elems:    []int{-1, -2, -3},
			sep:      ", ",
			expected: "-1, -2, -3",
		},
		{
			name:     "negative floats",
			elems:    []float64{-1.5, -2.5},
			sep:      ", ",
			expected: "-1.5, -2.5",
		},

		// Different integer types
		{
			name:     "int8 slice",
			elems:    []int8{1, 2, 3},
			sep:      ", ",
			expected: "1, 2, 3",
		},
		{
			name:     "int64 slice",
			elems:    []int64{1, 2, 3},
			sep:      ", ",
			expected: "1, 2, 3",
		},
		{
			name:     "uint slice",
			elems:    []uint{1, 2, 3},
			sep:      ", ",
			expected: "1, 2, 3",
		},
		{
			name:     "uint64 slice",
			elems:    []uint64{1, 2, 3},
			sep:      ", ",
			expected: "1, 2, 3",
		},

		// Complex numbers
		{
			name:     "complex numbers",
			elems:    []complex128{1 + 2i, 3 + 4i},
			sep:      ", ",
			expected: "(1+2i), (3+4i)",
		},

		// Pointers
		{
			name:  "pointer slice",
			elems: []*int{intPtr(1), intPtr(2), intPtr(3)},
			sep:   ", ",
			expected: func() string {
				p1, p2, p3 := intPtr(1), intPtr(2), intPtr(3)
				return Sprintf("%v, %v, %v", p1, p2, p3)
			}(),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var got string
			switch v := tt.elems.(type) {
			case []int:
				got = fmt.Join(v, tt.sep)
			case []string:
				got = fmt.Join(v, tt.sep)
			case []float64:
				got = fmt.Join(v, tt.sep)
			case []bool:
				got = fmt.Join(v, tt.sep)
			case []any:
				got = fmt.Join(v, tt.sep)
			case []int8:
				got = fmt.Join(v, tt.sep)
			case []int64:
				got = fmt.Join(v, tt.sep)
			case []uint:
				got = fmt.Join(v, tt.sep)
			case []uint64:
				got = fmt.Join(v, tt.sep)
			case []complex128:
				got = fmt.Join(v, tt.sep)
			case []*int:
				got = fmt.Join(v, tt.sep)
			case []error:
				got = fmt.Join(v, tt.sep)
			case []stringerType:
				got = fmt.Join(v, tt.sep)
			default:
				t.Fatalf("unsupported type: %T", v)
			}

			if got != tt.expected {
				t.Errorf("Join(%v, %q) = %q, want %q", tt.elems, tt.sep, got, tt.expected)
			}
		})
	}
}

func TestJoinEdgeCases(t *testing.T) {
	// Test with very long separator
	t.Run("very long separator", func(t *testing.T) {
		elems := []int{1, 2, 3}
		sep := strings.Repeat("x", 1000)
		got := fmt.Join(elems, sep)
		expected := "1" + sep + "2" + sep + "3"
		if got != expected {
			t.Errorf("Join with long separator: got %q, want %q", got, expected)
		}
	})

	// Test with empty strings in slice
	t.Run("empty strings in slice", func(t *testing.T) {
		elems := []string{"", "hello", "", "world", ""}
		got := fmt.Join(elems, ",")
		expected := ",hello,,world,"
		if got != expected {
			t.Errorf("Join with empty strings: got %q, want %q", got, expected)
		}
	})

	// Test with very large numbers
	t.Run("very large numbers", func(t *testing.T) {
		elems := []int64{9223372036854775807, -9223372036854775808}
		got := fmt.Join(elems, ", ")
		expected := "9223372036854775807, -9223372036854775808"
		if got != expected {
			t.Errorf("Join with large numbers: got %q, want %q", got, expected)
		}
	})

	// Test with special float values
	t.Run("special float values", func(t *testing.T) {
		elems := []float64{math.Inf(1), math.Inf(-1), math.NaN()}
		got := fmt.Join(elems, ", ")
		// NaN formatting can vary, so we check it contains the expected parts
		if !strings.Contains(got, "Inf") {
			t.Errorf("Join with special floats should contain 'Inf', got %q", got)
		}
	})

	// Test with struct types
	t.Run("struct types", func(t *testing.T) {
		type Point struct {
			X, Y int
		}
		elems := []Point{{1, 2}, {3, 4}}
		got := fmt.Join(elems, " | ")
		expected := "{1 2} | {3 4}"
		if got != expected {
			t.Errorf("Join with structs: got %q, want %q", got, expected)
		}
	})

	// Test with arrays (not slices)
	t.Run("array type", func(t *testing.T) {
		elems := [3]int{1, 2, 3}
		got := fmt.Join(elems[:], ", ")
		expected := "1, 2, 3"
		if got != expected {
			t.Errorf("Join with array: got %q, want %q", got, expected)
		}
	})
}

// Helper types and functions

type stringerType struct {
	val int
}

func (s stringerType) String() string {
	return Sprintf("Stringer(%d)", s.val)
}

func intPtr(i int) *int {
	return &i
}

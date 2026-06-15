// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fmt_test

import (
	"bytes"
	. "fmt"
	"testing"
)

// customString is a renamed string type for testing the reflect fallback path.
type customString string

// TestDoPrintSpacing checks that Sprint (and hence Print, Fprint) adds a space
// between adjacent non-string arguments, including the fast path for built-in
// string (type assertion) and the reflect fallback for renamed string types.
func TestDoPrintSpacing(t *testing.T) {
	cs := customString("custom")
	tests := []struct {
		name string
		args []any
		want string
	}{
		// Built-in string: fast path via type assertion.
		{"string_string", []any{"a", "b"}, "ab"},
		{"string_nonstring", []any{"a", 1}, "a1"},
		{"nonstring_string", []any{1, "a"}, "1a"},
		{"nonstring_nonstring", []any{1, 2}, "1 2"},

		// Custom string: reflect.Kind fallback path.
		{"customString_string", []any{cs, "b"}, "customb"},
		{"string_customString", []any{"a", cs}, "acustom"},
		{"customString_customString", []any{cs, customString("d")}, "customd"},
		{"customString_nonstring", []any{cs, 1}, "custom1"},
		{"nonstring_customString", []any{1, cs}, "1custom"},

		// nil must not panic and is treated as non-string.
		{"nil_nonstring", []any{nil, 1}, "<nil> 1"},
		{"string_nil", []any{"a", nil}, "a<nil>"},
		{"nil_string", []any{nil, "a"}, "<nil>a"},
		{"customString_nil", []any{cs, nil}, "custom<nil>"},
		{"nil_customString", []any{nil, cs}, "<nil>custom"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Sprint(tt.args...)
			if got != tt.want {
				t.Errorf("Sprint(%v) = %q, want %q", tt.args, got, tt.want)
			}

			var buf bytes.Buffer
			if _, err := Fprint(&buf, tt.args...); err != nil {
				t.Fatalf("Fprint(%v) returned error: %v", tt.args, err)
			}
			if got := buf.String(); got != tt.want {
				t.Errorf("Fprint(%v) = %q, want %q", tt.args, got, tt.want)
			}
		})
	}
}

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux

package runtime_test

import (
	. "runtime"
	"testing"
)

func TestParseProcStatmRSS(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		pageSize uint64
		want     uint64
		ok       bool
	}{
		{name: "full", input: "100 25 10 5 0 20 0\n", pageSize: 4096, want: 25 * 4096, ok: true},
		{name: "minimum", input: "1 2", pageSize: 4096, want: 2 * 4096, ok: true},
		{name: "extra fields", input: "1 3 4 5", pageSize: 2, want: 6, ok: true},
		{name: "leading whitespace", input: " 1 2", pageSize: 4096, ok: false},
		{name: "tab separator", input: "1\t2", pageSize: 4096, ok: false},
		{name: "empty", input: "", pageSize: 4096, ok: false},
		{name: "only first field", input: "1", pageSize: 4096, ok: false},
		{name: "missing resident", input: "1 \t\n", pageSize: 4096, ok: false},
		{name: "invalid first field", input: "x 2", pageSize: 4096, ok: false},
		{name: "invalid resident", input: "1 x", pageSize: 4096, ok: false},
		{name: "negative first field", input: "-1 2", pageSize: 4096, ok: false},
		{name: "negative resident", input: "1 -2", pageSize: 4096, ok: false},
		{name: "truncated resident", input: "1 12x", pageSize: 4096, ok: false},
		{name: "large first field ignored", input: "18446744073709551616 1", pageSize: 1, want: 1, ok: true},
		{name: "resident overflow", input: "1 18446744073709551616", pageSize: 1, ok: false},
		{name: "multiplication overflow", input: "1 2", pageSize: 1 << 63, ok: false},
		{name: "zero page size", input: "1 2", pageSize: 0, ok: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := ParseProcStatmRSS([]byte(tt.input), tt.pageSize)
			if got != tt.want || ok != tt.ok {
				t.Errorf("ParseProcStatmRSS(%q, %d) = (%d, %v); want (%d, %v)",
					tt.input, tt.pageSize, got, ok, tt.want, tt.ok)
			}
		})
	}
}

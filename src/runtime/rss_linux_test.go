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
		name       string
		input      string
		pageSize   uint64
		wantRSS    uint64
		wantShared uint64
		ok         bool
	}{
		{name: "full", input: "100 25 10 5 0 20 0\n", pageSize: 4096, wantRSS: 25 * 4096, wantShared: 10 * 4096, ok: true},
		{name: "minimum", input: "3 2 0 ", pageSize: 4096, wantRSS: 2 * 4096, ok: true},
		{name: "extra fields", input: "5 3 2 5", pageSize: 2, wantRSS: 6, wantShared: 4, ok: true},
		{name: "max width", input: "18446744073709551615 18446744073709551615 18446744073709551615 ", pageSize: 1, wantRSS: ^uint64(0), wantShared: ^uint64(0), ok: true},
		{name: "leading whitespace", input: " 3 2 1 ", pageSize: 4096},
		{name: "tab separator", input: "3\t2 1 ", pageSize: 4096},
		{name: "empty", input: "", pageSize: 4096},
		{name: "only first field", input: "3", pageSize: 4096},
		{name: "missing resident", input: "3  1 ", pageSize: 4096},
		{name: "missing shared", input: "3 2 ", pageSize: 4096},
		{name: "invalid first field", input: "x 2 1 ", pageSize: 4096},
		{name: "invalid resident", input: "3 2x 1 ", pageSize: 4096},
		{name: "invalid shared", input: "3 2 1x ", pageSize: 4096},
		{name: "negative first field", input: "-3 2 1 ", pageSize: 4096},
		{name: "negative resident", input: "3 -2 1 ", pageSize: 4096},
		{name: "negative shared", input: "3 2 -1 ", pageSize: 4096},
		{name: "truncated shared", input: "3 2 1", pageSize: 4096},
		{name: "large first field ignored", input: "18446744073709551616 1 1 ", pageSize: 1, wantRSS: 1, wantShared: 1, ok: true},
		{name: "resident overflow", input: "1 18446744073709551616 1 ", pageSize: 1},
		{name: "shared overflow", input: "1 1 18446744073709551616 ", pageSize: 1},
		{name: "shared exceeds RSS", input: "3 1 2 ", pageSize: 4096},
		{name: "multiplication overflow", input: "3 2 1 ", pageSize: 1 << 63},
		{name: "zero page size", input: "3 2 1 ", pageSize: 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rss, shared, ok := ParseProcStatmRSS([]byte(tt.input), tt.pageSize)
			if rss != tt.wantRSS || shared != tt.wantShared || ok != tt.ok {
				t.Errorf("ParseProcStatmRSS(%q, %d) = (%d, %d, %v); want (%d, %d, %v)",
					tt.input, tt.pageSize, rss, shared, ok, tt.wantRSS, tt.wantShared, tt.ok)
			}
		})
	}
}

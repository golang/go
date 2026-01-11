package ssa

import "testing"

func TestSymbolMatcher(t *testing.T) {
	m := newSymbolMatcher([]string{"log.error", "pkg.sub.*"})

	cases := []struct {
		sym     string
		matches bool
	}{
		{"log.error", true},
		{"log.errorx", false},
		{"pkg.sub.fn", true},
		{"pkg.subpkg.fn", true},
		{"pkg.other.fn", false},
	}

	for _, tc := range cases {
		if got := m.match(tc.sym); got != tc.matches {
			t.Fatalf("match(%q)=%v want %v", tc.sym, got, tc.matches)
		}
	}
}

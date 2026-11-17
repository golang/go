// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package settings

import (
	"fmt"
	"strings"
	"testing"
)

func TestScalarOptimizationScope(t *testing.T) {
	empty := logosModule(t, "")
	o3 := logosModule(t, "[build]\nopt_level = \"O3\"\n")
	for _, logos := range []bool{false, true} {
		for _, level := range []string{"", "0", "1", "2", "3", "s"} {
			t.Run(fmt.Sprintf("logos=%t/level=%s", logos, level), func(t *testing.T) {
				s, err := Load(empty, Flags{Logos: logos, Opt: level})
				if err != nil {
					t.Fatal(err)
				}
				if got := strings.Contains(strings.Join(s.Gcflags, " "), "logosopt=1"); got != (level == "3") {
					t.Fatalf("flags = %q: scalar analysis enabled = %t, want %t", s.Gcflags, got, level == "3")
				}
			})
		}
	}
	for _, tc := range []struct {
		name  string
		flags Flags
		want  bool
	}{
		{"logos_configuration", Flags{Logos: true}, true},
		{"plain_go_ignores_configuration", Flags{}, false},
		{"explicit_lower_tier_wins", Flags{Logos: true, Opt: "2"}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s, err := Load(o3, tc.flags)
			if err != nil {
				t.Fatal(err)
			}
			if got := strings.Contains(strings.Join(s.Gcflags, " "), "logosopt=1"); got != tc.want {
				t.Fatalf("flags = %q: scalar analysis enabled = %t, want %t", s.Gcflags, got, tc.want)
			}
		})
	}
}

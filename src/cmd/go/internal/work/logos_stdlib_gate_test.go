// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package work

import (
	"slices"
	"strings"
	"testing"

	"cmd/go/internal/load"
	"cmd/go/internal/logos/settings"
)

func variableMakeKeys(flags []string) []string {
	var keys []string
	for _, f := range flags {
		for k := range strings.SplitSeq(strings.TrimPrefix(f, logosDebugFlagPrefix), logosDebugKeySep) {
			if strings.HasPrefix(k, "variablemakethreshold=") {
				keys = append(keys, k)
			}
		}
	}
	return keys
}

func TestLogosStandardProvenanceKeepsDefaultVariableMake(t *testing.T) {
	old := forcedGcflags
	t.Cleanup(func() { forcedGcflags = old })

	standard := &load.Package{PackagePublic: load.PackagePublic{Standard: true}}
	standardExternalTest := &load.Package{PackagePublic: load.PackagePublic{ForTest: "fmt"}}
	userPackage := &load.Package{}
	userExternalTest := &load.Package{PackagePublic: load.PackagePublic{ForTest: "example.com/m"}}

	for _, tc := range []struct {
		level string
		want  string
	}{
		{"2", "variablemakethreshold=256"},
		{"3", "variablemakethreshold=4096"},
	} {
		s, err := settings.Load("", settings.Flags{Opt: tc.level})
		if err != nil {
			t.Fatalf("Load(Opt=%q) = %v, want no error", tc.level, err)
		}
		forcedGcflags = s.Gcflags

		for name, p := range map[string]*load.Package{"std": standard, "std external test": standardExternalTest} {
			got := logosForcedGcflags(p)
			if keys := variableMakeKeys(got); len(keys) != 0 {
				t.Errorf("O%s %s: forced flags %q carry %q, want the default 32-byte buffer", tc.level, name, got, keys)
			}
			if len(got) != len(forcedGcflags) || !strings.Contains(strings.Join(got, " "), "inlbudget=") {
				t.Errorf("O%s %s: forced flags %q lost the level's other keys (level flags %q)", tc.level, name, got, forcedGcflags)
			}
		}

		for name, p := range map[string]*load.Package{"user": userPackage, "user external test": userExternalTest} {
			got := logosForcedGcflags(p)
			if keys := variableMakeKeys(got); !slices.Equal(keys, []string{tc.want}) {
				t.Errorf("O%s %s: forced flags %q carry %q, want %q", tc.level, name, got, keys, tc.want)
			}
		}
	}
}

func TestLogosDropVariableMakeKeepsOtherKeys(t *testing.T) {
	for _, tc := range []struct{ in, want string }{
		{"-d=inlbudget=120,variablemakethreshold=256", "-d=inlbudget=120"},
		{"-d=variablemakethreshold=4096,inlbudget=400", "-d=inlbudget=400"},
		{"-d=inlbudget=400,inlrecursivebudget=80,variablemakethreshold=4096", "-d=inlbudget=400,inlrecursivebudget=80"},
		{"-d=inlbudget=15", "-d=inlbudget=15"},
		{"-N", "-N"},
		{"-d=variablemakethreshold=256", ""},
	} {
		if got := logosDropVariableMake(tc.in); got != tc.want {
			t.Errorf("logosDropVariableMake(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package config

import "testing"

// M19-S1: the [build] gc key is stored as written and survives Format; the
// mode check happens at build time with the text of the -gc flag.

func TestParseGC(t *testing.T) {
	for _, tc := range []struct {
		name    string
		data    string
		gc      string
		set     bool
		wantErr string
	}{
		{"absent", "[build]\ncgo = \"allow\"\n", "", false, ""},
		{"batch", "[build]\ngc = \"batch\"\n", "batch", true, ""},
		{"literal string", "[build]\ngc = 'epsilon'\n", "epsilon", true, ""},
		{"empty", "[build]\ngc = \"\"\n", "", true, ""},
		{"unknown mode is kept", "[build]\ngc = \"x\"\n", "x", true, ""},
		{"root key rejected", "gc = \"batch\"\n", "", false, `logos.toml:1:1: unknown key "gc"`},
		{"misplaced key", "[project]\ngc = \"batch\"\n", "", false, `logos.toml:2:1: unknown key "gc"`},
		{"duplicate key", "[build]\ngc = \"batch\"\ngc = \"zgc\"\n", "", false, `logos.toml:3:1: duplicate key "gc"`},
		{"array rejected", "[build]\ngc = [\"batch\"]\n", "", false, `logos.toml:2:6: want string for key "gc"`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f, err := Parse("logos.toml", []byte(tc.data))
			if tc.wantErr != "" {
				if err == nil || err.Error() != tc.wantErr {
					t.Fatalf("Parse error = %v, want %q", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if f.Build.GC != tc.gc || f.Build.GCSet != tc.set {
				t.Errorf("Build.GC = %q (set %v), want %q (set %v)", f.Build.GC, f.Build.GCSet, tc.gc, tc.set)
			}
			again, err := Parse("logos.toml", f.Format())
			if err != nil {
				t.Fatalf("Format output does not parse: %v\n%s", err, f.Format())
			}
			if again.Build.GC != f.Build.GC || again.Build.GCSet != f.Build.GCSet {
				t.Errorf("round trip: Build.GC = %q (set %v), want %q (set %v)", again.Build.GC, again.Build.GCSet, f.Build.GC, f.Build.GCSet)
			}
		})
	}
}

func TestGCModes(t *testing.T) {
	want := "greentea zgc immix wasm batch epsilon"
	got := ""
	for i, m := range GCModes {
		if i > 0 {
			got += " "
		}
		got += m
	}
	if got != want {
		t.Errorf("GCModes = %q, want %q", got, want)
	}
}

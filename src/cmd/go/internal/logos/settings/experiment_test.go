// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package settings

import "testing"

func TestPreemptibleUserState(t *testing.T) {
	tests := []struct {
		raw               string
		mentioned, enable bool
	}{
		{"", false, false},
		{"nojsonv2", false, false},
		{"preemptibleloops", true, true},
		{"nopreemptibleloops", true, false},
		{"none", true, false},
		{"none,preemptibleloops", true, true},
		{"preemptibleloops,none", true, false},
		{"nojsonv2,nopreemptibleloops", true, false},
	}
	for _, tt := range tests {
		mentioned, enabled := preemptibleUserState(tt.raw)
		if mentioned != tt.mentioned || enabled != tt.enable {
			t.Errorf("preemptibleUserState(%q) = (%v, %v), want (%v, %v)",
				tt.raw, mentioned, enabled, tt.mentioned, tt.enable)
		}
	}
}

func TestPreemptibleDefault(t *testing.T) {
	tests := []struct {
		name         string
		raw, startup string
		has, logos   bool
		goarch       string
		wantRaw      string
		wantApply    bool
	}{
		{"wasmDefaultApplied", "", "", false, true, "wasm", "preemptibleloops", true},
		{"plainGoNoDefault", "", "", false, false, "wasm", "", false},
		{"nonWasmNoDefault", "", "", false, true, "amd64", "", false},
		{"userOptOutKept", "", "nopreemptibleloops", false, true, "wasm", "", false},
		{"userNoneKept", "", "none", false, true, "wasm", "", false},
		{"explicitEnableKept", "preemptibleloops", "preemptibleloops", true, true, "wasm", "", false},
		{"composesWithUserExperiment", "nojsonv2", "nojsonv2", false, true, "wasm", "nojsonv2,preemptibleloops", true},
		{"stripsOnRetarget", "preemptibleloops", "", true, true, "amd64", "", true},
		{"retargetKeepsOtherTokens", "nojsonv2,preemptibleloops", "nojsonv2", true, true, "amd64", "nojsonv2", true},
		{"retargetKeepsExplicit", "preemptibleloops", "preemptibleloops", true, true, "amd64", "", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotRaw, apply := PreemptibleDefault(tt.raw, tt.startup, tt.has, tt.logos, tt.goarch)
			if gotRaw != tt.wantRaw || apply != tt.wantApply {
				t.Errorf("PreemptibleDefault(%q, %q, %v, %v, %q) = (%q, %v), want (%q, %v)",
					tt.raw, tt.startup, tt.has, tt.logos, tt.goarch, gotRaw, apply, tt.wantRaw, tt.wantApply)
			}
		})
	}
}

func TestWithArena(t *testing.T) {
	tests := []struct {
		name, raw, want string
	}{
		{"emptyBase", "", "logosarena"},
		{"keepsUserExperiments", "nojsonv2", "nojsonv2,logosarena"},
		{"reappendStaysEnabled", "logosarena", "logosarena,logosarena"},
		{"winsOverEarlierNegation", "nologosarena", "nologosarena,logosarena"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := WithArena(tt.raw); got != tt.want {
				t.Errorf("WithArena(%q) = %q, want %q", tt.raw, got, tt.want)
			}
		})
	}
}

func TestDropLogosExperiments(t *testing.T) {
	tests := []struct {
		name, raw, want string
	}{
		{"dropsArena", "jsonv2,logosarena", "jsonv2"},
		{"dropsNegated", "nologosarena,jsonv2", "jsonv2"},
		{"dropsAll", "logosarena", ""},
		{"keepsPreemptible", "preemptibleloops", "preemptibleloops"},
		{"keepsPlain", "jsonv2", "jsonv2"},
		{"empty", "", ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := DropLogosExperiments(tt.raw); got != tt.want {
				t.Errorf("DropLogosExperiments(%q) = %q, want %q", tt.raw, got, tt.want)
			}
		})
	}
}

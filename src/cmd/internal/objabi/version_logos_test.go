// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package objabi

import "testing"

func TestStripExperiment(t *testing.T) {
	tests := []struct {
		name    string
		version string
		want    string
	}{
		{"plain release", "go1.27.2", "go1.27.2"},
		{"plain minor", "go1.26", "go1.26"},
		{"one experiment dash", "go1.27.2-X:logosfastcgo", "go1.27.2"},
		{"one experiment space", "go1.27.2 X:fieldtrack", "go1.27.2"},
		{"multiple experiments dash", "go1.27.2-X:logosfastcgo,regabi", "go1.27.2"},
		{"multiple experiments space", "go1.27.2 X:fieldtrack,synctest,greenteagc", "go1.27.2"},
		{"devel plain", "devel go1.27-d729a8a", "devel go1.27-d729a8a"},
		{"devel date plain", "devel go1.27-d729a8a Fri Jan 1 00:00:00 2026 +0000", "devel go1.27-d729a8a Fri Jan 1 00:00:00 2026 +0000"},
		{"devel one experiment space", "devel go1.27-d729a8a X:logosfastcgo", "devel go1.27-d729a8a"},
		{"devel one experiment dash", "devel go1.27-d729a8a-X:logosfastcgo", "devel go1.27-d729a8a"},
		{"devel multiple experiments space", "devel go1.27-d729a8a X:logosfastcgo,regabi", "devel go1.27-d729a8a"},
		{"devel date multiple experiments", "devel go1.27-d729a8a Fri Jan 1 00:00:00 2026 +0000 X:fieldtrack,synctest", "devel go1.27-d729a8a Fri Jan 1 00:00:00 2026 +0000"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := StripExperiment(tt.version); got != tt.want {
				t.Errorf("StripExperiment(%q) = %q, want %q", tt.version, got, tt.want)
			}
		})
	}
}

func TestStripExperimentComparison(t *testing.T) {
	tests := []struct {
		name string
		v1   string
		v2   string
		want bool
	}{
		{"plain vs logosfastcgo", "go1.27.2", "go1.27.2-X:logosfastcgo", true},
		{"logosfastcgo vs plain", "go1.27.2-X:logosfastcgo", "go1.27.2", true},
		{"different experiments", "go1.27.2-X:logosfastcgo", "go1.27.2-X:fieldtrack", true},
		{"space vs dash experiments", "go1.27.2 X:fieldtrack", "go1.27.2-X:logosfastcgo,regabi", true},
		{"devel vs devel experiment", "devel go1.27-d729a8a X:logosfastcgo", "devel go1.27-d729a8a", true},
		{"devel different experiments", "devel go1.27-d729a8a X:fieldtrack", "devel go1.27-d729a8a X:regabi", true},
		{"different patch versions", "go1.27.2", "go1.27.1", false},
		{"different patch with experiment", "go1.27.2", "go1.27.1-X:logosfastcgo", false},
		{"both different with same experiment", "go1.27.2-X:logosfastcgo", "go1.27.1-X:logosfastcgo", false},
		{"minor version mismatch", "go1.27.2", "go1.28.0", false},
		{"release vs devel", "go1.27.2", "devel go1.27-d729a8a", false},
		{"different devel commits", "devel go1.27-commitA", "devel go1.27-commitB X:logosfastcgo", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := StripExperiment(tt.v1) == StripExperiment(tt.v2)
			if got != tt.want {
				t.Errorf("StripExperiment(%q) == StripExperiment(%q): got %v, want %v", tt.v1, tt.v2, got, tt.want)
			}
		})
	}
}

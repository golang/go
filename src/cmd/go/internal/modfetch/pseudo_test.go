// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modfetch

import (
	"testing"
	"time"
)

var pseudoTests = []struct {
	major   string
	older   string
	version string
}{
	{"", "", "v0.0.0-20060102150405-hash"},
	{"v0", "", "v0.0.0-20060102150405-hash"},
	{"v1", "", "v1.0.0-20060102150405-hash"},
	{"v2", "", "v2.0.0-20060102150405-hash"},
	{"unused", "v0.0.0", "v0.0.1-0.20060102150405-hash"},
	{"unused", "v1.2.3", "v1.2.4-0.20060102150405-hash"},
	{"unused", "v1.2.99999999999999999", "v1.2.100000000000000000-0.20060102150405-hash"},
	{"unused", "v1.2.3-pre", "v1.2.3-pre.0.20060102150405-hash"},
	{"unused", "v1.3.0-pre", "v1.3.0-pre.0.20060102150405-hash"},
}

var pseudoTime = time.Date(2006, 1, 2, 15, 4, 5, 0, time.UTC)

func TestPseudoVersion(t *testing.T) {
	for _, tt := range pseudoTests {
		v := PseudoVersion(tt.major, tt.older, pseudoTime, "hash")
		if v != tt.version {
			t.Errorf("PseudoVersion(%q, %q, ...) = %v, want %v", tt.major, tt.older, v, tt.version)
		}
	}
}

func TestIsPseudoVersion(t *testing.T) {
	for _, tt := range pseudoTests {
		if !IsPseudoVersion(tt.version) {
			t.Errorf("IsPseudoVersion(%q) = false, want true", tt.version)
		}
		if IsPseudoVersion(tt.older) {
			t.Errorf("IsPseudoVersion(%q) = true, want false", tt.older)
		}
	}
}

func TestPseudoVersionTime(t *testing.T) {
	for _, tt := range pseudoTests {
		tm, err := PseudoVersionTime(tt.version)
		if tm != pseudoTime || err != nil {
			t.Errorf("PseudoVersionTime(%q) = %v, %v, want %v, nil", tt.version, tm.Format(time.RFC3339), err, pseudoTime.Format(time.RFC3339))
		}
		tm, err = PseudoVersionTime(tt.older)
		if tm != (time.Time{}) || err == nil {
			t.Errorf("PseudoVersionTime(%q) = %v, %v, want %v, error", tt.older, tm.Format(time.RFC3339), err, time.Time{}.Format(time.RFC3339))
		}
	}
}

func TestPseudoVersionRev(t *testing.T) {
	for _, tt := range pseudoTests {
		rev, err := PseudoVersionRev(tt.version)
		if rev != "hash" || err != nil {
			t.Errorf("PseudoVersionRev(%q) = %q, %v, want %q, nil", tt.older, rev, err, "hash")
		}
		rev, err = PseudoVersionRev(tt.older)
		if rev != "" || err == nil {
			t.Errorf("PseudoVersionRev(%q) = %q, %v, want %q, error", tt.older, rev, err, "")
		}
	}
}

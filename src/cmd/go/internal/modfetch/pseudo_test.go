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
	{"unused", "v0.0.0--", "v0.0.0--.0.20060102150405-hash"},
	{"unused", "v1.0.0+metadata", "v1.0.1-0.20060102150405-hash+metadata"},
	{"unused", "v2.0.0+incompatible", "v2.0.1-0.20060102150405-hash+incompatible"},
	{"unused", "v2.3.0-pre+incompatible", "v2.3.0-pre.0.20060102150405-hash+incompatible"},
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

func TestInvalidPseudoVersionTime(t *testing.T) {
	const v = "---"
	if _, err := PseudoVersionTime(v); err == nil {
		t.Error("expected error, got nil instead")
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

func TestPseudoVersionBase(t *testing.T) {
	for _, tt := range pseudoTests {
		base, err := PseudoVersionBase(tt.version)
		if err != nil {
			t.Errorf("PseudoVersionBase(%q): %v", tt.version, err)
		} else if base != tt.older {
			t.Errorf("PseudoVersionBase(%q) = %q; want %q", tt.version, base, tt.older)
		}
	}
}

func TestInvalidPseudoVersionBase(t *testing.T) {
	for _, in := range []string{
		"v0.0.0",
		"v0.0.0-",                                 // malformed: empty prerelease
		"v0.0.0-0.20060102150405-hash",            // Z+1 == 0
		"v0.1.0-0.20060102150405-hash",            // Z+1 == 0
		"v1.0.0-0.20060102150405-hash",            // Z+1 == 0
		"v0.0.0-20060102150405-hash+incompatible", // "+incompatible without base version
		"v0.0.0-20060102150405-hash+metadata",     // other metadata without base version
	} {
		base, err := PseudoVersionBase(in)
		if err == nil || base != "" {
			t.Errorf(`PseudoVersionBase(%q) = %q, %v; want "", error`, in, base, err)
		}
	}
}

func TestIncDecimal(t *testing.T) {
	cases := []struct {
		in, want string
	}{
		{"0", "1"},
		{"1", "2"},
		{"99", "100"},
		{"100", "101"},
		{"101", "102"},
	}

	for _, tc := range cases {
		got := incDecimal(tc.in)
		if got != tc.want {
			t.Fatalf("incDecimal(%q) = %q; want %q", tc.in, tc.want, got)
		}
	}
}

func TestDecDecimal(t *testing.T) {
	cases := []struct {
		in, want string
	}{
		{"", ""},
		{"0", ""},
		{"00", ""},
		{"1", "0"},
		{"2", "1"},
		{"99", "98"},
		{"100", "99"},
		{"101", "100"},
	}

	for _, tc := range cases {
		got := decDecimal(tc.in)
		if got != tc.want {
			t.Fatalf("decDecimal(%q) = %q; want %q", tc.in, tc.want, got)
		}
	}
}

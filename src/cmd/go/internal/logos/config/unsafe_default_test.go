// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package config

import (
	"strings"
	"testing"
)

func TestUnsafeCapabilityDefaultsAndFormat(t *testing.T) {
	for _, tc := range []struct {
		name, source, want string
		written            bool
	}{
		{"missing table", "[build]\ncgo = \"allow\"\n", "allow", false},
		{"missing key", "[capabilities]\ntransitive_net = \"warn\"\n", "allow", false},
		{"allow", "[capabilities]\ntransitive_unsafe = \"allow\"\n", "allow", false},
		{"warn", "[capabilities]\ntransitive_unsafe = \"warn\"\n", "warn", true},
		{"deny", "[capabilities]\ntransitive_unsafe = \"deny\"\n", "deny", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f, err := Parse("logos.toml", []byte(tc.source))
			if err != nil {
				t.Fatal(err)
			}
			if got := f.Capabilities.TransitiveUnsafe; got != tc.want {
				t.Fatalf("severity = %q, want %q", got, tc.want)
			}
			formatted := string(f.Format())
			if got := strings.Contains(formatted, "transitive_unsafe ="); got != tc.written {
				t.Fatalf("formatted = %q, key present = %v, want %v", formatted, got, tc.written)
			}
			roundtrip, err := Parse("logos.toml", []byte(formatted))
			if err != nil {
				t.Fatal(err)
			}
			if got := roundtrip.Capabilities.TransitiveUnsafe; got != tc.want {
				t.Fatalf("round-trip severity = %q, want %q", got, tc.want)
			}
		})
	}
	f, err := Load(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if got := f.Capabilities.TransitiveUnsafe; got != "allow" {
		t.Fatalf("missing file severity = %q, want allow", got)
	}
}

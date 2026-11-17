// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"flag"
	"testing"
)

func TestLogosCapExternalTestOwner(t *testing.T) {
	tests := []struct {
		name, pkg, origin string
		mains, deps       []string
		std               map[string]bool
		testBinary, want  bool
	}{
		{
			name: "main module root", pkg: "example.com/app_test", origin: "example.com/app",
			mains: []string{"example.com/app"}, testBinary: true, want: true,
		},
		{
			name: "main subpackage", pkg: "example.com/app/sub_test", origin: "example.com/app/sub",
			mains: []string{"example.com/app"}, testBinary: true, want: true,
		},
		{
			name: "nested dependency root", pkg: "example.com/app/nested_test", origin: "example.com/app/nested",
			mains: []string{"example.com/app"}, deps: []string{"example.com/app/nested"}, testBinary: true,
		},
		{
			name: "ordinary dependency", pkg: "example.com/dep_test", origin: "example.com/dep",
			mains: []string{"example.com/app"}, deps: []string{"example.com/dep"}, testBinary: true,
		},
		{
			name: "nested workspace main", pkg: "example.com/app/nested_test", origin: "example.com/app/nested",
			mains: []string{"example.com/app", "example.com/app/nested"}, testBinary: true, want: true,
		},
		{
			name: "longest workspace main", pkg: "example.com/app/nested/deep_test", origin: "example.com/app/nested/deep",
			mains: []string{"example.com/app", "example.com/app/nested/deep"}, deps: []string{"example.com/app/nested"}, testBinary: true, want: true,
		},
		{
			name: "root package needs provenance", pkg: "example.com/app_test",
			mains: []string{"example.com/app"}, testBinary: true,
		},
		{
			name: "ordinary suffixed dependency", pkg: "example.com/app_test", origin: "example.com/unrelated",
			mains: []string{"example.com/app"}, deps: []string{"example.com/app_test"}, testBinary: true,
		},
		{
			name: "ordinary suffixed main package", pkg: "example.com/app/nested_test", origin: "example.com/unrelated",
			mains: []string{"example.com/app"}, deps: []string{"example.com/app/nested"}, testBinary: true, want: true,
		},
		{
			name: "provenance is exact", pkg: "example.com/app_test/sub", origin: "example.com/app",
			mains: []string{"example.com/app"}, deps: []string{"example.com/app_test"}, testBinary: true,
		},
		{
			name: "longer suffix", pkg: "example.com/app_tester", origin: "example.com/app",
			mains: []string{"example.com/app"}, testBinary: true,
		},
		{
			name: "ordinary binary ignores provenance", pkg: "example.com/app_test", origin: "example.com/app",
			mains: []string{"example.com/app"},
		},
		{
			name: "standard external test", pkg: "fmt_test", origin: "fmt",
			std: map[string]bool{"fmt_test": true}, testBinary: true, want: true,
		},
		{
			name: "command line external test", pkg: "command-line-arguments_test", origin: "command-line-arguments",
			testBinary: true, want: true,
		},
		{
			name: "internal test stays main", pkg: "example.com/app", origin: "example.com/app",
			mains: []string{"example.com/app"}, testBinary: true, want: true,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			old, had := strdata["testing.testBinary"]
			if tc.testBinary {
				strdata["testing.testBinary"] = "1"
			} else {
				delete(strdata, "testing.testBinary")
			}
			t.Cleanup(func() {
				if had {
					strdata["testing.testBinary"] = old
				} else {
					delete(strdata, "testing.testBinary")
				}
			})
			// The baseline has no origin flag. Keeping its unmodified
			// classification exposes the real wrong-ownership cases.
			if f := flag.Lookup("logos.test-package"); f != nil {
				previous := f.Value.String()
				if err := f.Value.Set(tc.origin); err != nil {
					t.Fatal(err)
				}
				t.Cleanup(func() {
					if err := f.Value.Set(previous); err != nil {
						t.Error(err)
					}
				})
			}
			a := &logosCapAuditor{
				policy: &logosCapPolicy{MainModules: tc.mains},
				deps:   tc.deps,
				std:    tc.std,
			}
			if got := a.exemptPkg(tc.pkg); got != tc.want {
				t.Errorf("exemptPkg(%q), origin %q = %v, want %v", tc.pkg, tc.origin, got, tc.want)
			}
		})
	}
}

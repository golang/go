// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package list

import (
	"reflect"
	"testing"
)

func TestPackageImportMap(t *testing.T) {
	tests := []struct {
		name            string
		imports         []string
		rawImports      []string
		compiledImports []string
		want            map[string]string
	}{
		{
			name:       "raw import",
			imports:    []string{"vendor/foo", "bar"},
			rawImports: []string{"foo", "bar"},
			want:       map[string]string{"foo": "vendor/foo"},
		},
		{
			name:            "compiled import",
			imports:         []string{"sync [runtime.test]", "runtime/cgo [runtime.test]"},
			rawImports:      []string{"sync"},
			compiledImports: []string{"runtime/cgo"},
			want: map[string]string{
				"sync":        "sync [runtime.test]",
				"runtime/cgo": "runtime/cgo [runtime.test]",
			},
		},
		{
			name:            "synthetic import before compiled imports",
			imports:         []string{"resolved/a", "runtime", "runtime/cgo [runtime.test]"},
			rawImports:      []string{"a"},
			compiledImports: []string{"runtime/cgo"},
			want: map[string]string{
				"a":           "resolved/a",
				"runtime/cgo": "runtime/cgo [runtime.test]",
			},
		},
		{
			name:       "synthetic import without compiled imports",
			imports:    []string{"testing", "runtime"},
			rawImports: []string{"testing"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := packageImportMap(tt.imports, tt.rawImports, tt.compiledImports)
			if !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("packageImportMap(%q, %q, %q) = %v; want %v", tt.imports, tt.rawImports, tt.compiledImports, got, tt.want)
			}
		})
	}
}

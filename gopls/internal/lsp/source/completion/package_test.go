// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package completion

import (
	"testing"

	"golang.org/x/tools/gopls/internal/lsp/source"
)

func TestIsValidDirName(t *testing.T) {
	tests := []struct {
		dirName string
		valid   bool
	}{
		{dirName: "", valid: false},
		//
		{dirName: "a", valid: true},
		{dirName: "abcdef", valid: true},
		{dirName: "AbCdEf", valid: true},
		//
		{dirName: "1a35", valid: true},
		{dirName: "a16", valid: true},
		//
		{dirName: "_a", valid: true},
		{dirName: "a_", valid: true},
		//
		{dirName: "~a", valid: false},
		{dirName: "a~", valid: true},
		//
		{dirName: "-a", valid: false},
		{dirName: "a-", valid: true},
		//
		{dirName: ".a", valid: false},
		{dirName: "a.", valid: false},
		//
		{dirName: "a~_b--c.-e", valid: true},
		{dirName: "~a~_b--c.-e", valid: false},
		{dirName: "a~_b--c.-e--~", valid: true},
		{dirName: "a~_b--2134dc42.-e6--~", valid: true},
		{dirName: "abc`def", valid: false},
		{dirName: "тест", valid: false},
		{dirName: "你好", valid: false},
	}
	for _, tt := range tests {
		valid := isValidDirName(tt.dirName)
		if tt.valid != valid {
			t.Errorf("%s: expected %v, got %v", tt.dirName, tt.valid, valid)
		}
	}
}

func TestConvertDirNameToPkgName(t *testing.T) {
	tests := []struct {
		dirName string
		pkgName source.PackageName
	}{
		{dirName: "a", pkgName: "a"},
		{dirName: "abcdef", pkgName: "abcdef"},
		{dirName: "AbCdEf", pkgName: "abcdef"},
		{dirName: "1a35", pkgName: "a35"},
		{dirName: "14a35", pkgName: "a35"},
		{dirName: "a16", pkgName: "a16"},
		{dirName: "_a", pkgName: "a"},
		{dirName: "a_", pkgName: "a"},
		{dirName: "a~", pkgName: "a"},
		{dirName: "a-", pkgName: "a"},
		{dirName: "a~_b--c.-e", pkgName: "abce"},
		{dirName: "a~_b--c.-e--~", pkgName: "abce"},
		{dirName: "a~_b--2134dc42.-e6--~", pkgName: "ab2134dc42e6"},
	}
	for _, tt := range tests {
		pkgName := convertDirNameToPkgName(tt.dirName)
		if tt.pkgName != pkgName {
			t.Errorf("%s: expected %v, got %v", tt.dirName, tt.pkgName, pkgName)
			continue
		}
	}
}

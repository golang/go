// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gover

import (
	"slices"
	"strings"
	"testing"

	"golang.org/x/mod/module"
)

func TestIsToolchain(t *testing.T) { test1(t, isToolchainTests, "IsToolchain", IsToolchain) }

var isToolchainTests = []testCase1[string, bool]{
	{"go", true},
	{"toolchain", true},
	{"anything", false},
	{"golang.org/toolchain", false},
}

func TestModCompare(t *testing.T) { test3(t, modCompareTests, "ModCompare", ModCompare) }

var modCompareTests = []testCase3[string, string, string, int]{
	{"go", "1.2", "1.3", -1},
	{"go", "v1.2", "v1.3", 0}, // equal because invalid
	{"go", "1.2", "1.2", 0},
	{"toolchain", "go1.2", "go1.3", -1},
	{"toolchain", "go1.2", "go1.2", 0},
	{"toolchain", "1.2", "1.3", -1},  // accepted but non-standard
	{"toolchain", "v1.2", "v1.3", 0}, // equal because invalid
	{"rsc.io/quote", "v1.2", "v1.3", -1},
	{"rsc.io/quote", "1.2", "1.3", 0}, // equal because invalid
}

func TestModIsValid(t *testing.T) { test2(t, modIsValidTests, "ModIsValid", ModIsValid) }

var modIsValidTests = []testCase2[string, string, bool]{
	{"go", "1.2", true},
	{"go", "v1.2", false},
	{"toolchain", "go1.2", true},
	{"toolchain", "v1.2", false},
	{"rsc.io/quote", "v1.2", true},
	{"rsc.io/quote", "1.2", false},
}

func TestModSort(t *testing.T) {
	test1(t, modSortTests, "ModSort", func(list []module.Version) []module.Version {
		out := slices.Clone(list)
		ModSort(out)
		return out
	})
}

var modSortTests = []testCase1[[]module.Version, []module.Version]{
	{
		mvl(`z v1.1; a v1.2; a v1.1; go 1.3; toolchain 1.3; toolchain 1.2; go 1.2`),
		mvl(`a v1.1; a v1.2; go 1.2; go 1.3; toolchain 1.2; toolchain 1.3; z v1.1`),
	},
}

func mvl(s string) []module.Version {
	var list []module.Version
	for _, f := range strings.Split(s, ";") {
		f = strings.TrimSpace(f)
		path, vers, _ := strings.Cut(f, " ")
		list = append(list, module.Version{Path: path, Version: vers})
	}
	return list
}

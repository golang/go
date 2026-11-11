// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"cmd/internal/sys"
	"slices"
	"strings"
	"testing"
)

func TestCgoRpathParser(t *testing.T) {
	tests := []struct {
		flags []string
		want  []string
	}{
		{[]string{"-Wl,-rpath,/a"}, []string{"/a"}},
		{[]string{"-Wl,-rpath=/a"}, []string{"/a"}},
		{[]string{"-Wl,--rpath=/a"}, []string{"/a"}},
		{[]string{"-Wl,-rpath,/a,-rpath,/b"}, []string{"/a", "/b"}},
		{[]string{"-Wl,-rpath,/a,--enable-new-dtags"}, []string{"/a"}},
		{[]string{"-Wl,--enable-new-dtags,-rpath,/a"}, []string{"/a"}},
		{[]string{"-Wl,-rpath", "-Wl,/a"}, []string{"/a"}},
		{[]string{"-Wl,-R,/a", "-Wl,-R=/b", "-Wl,-R/c"}, []string{"/a", "/b", "/c"}},
		{[]string{"-Wl,-R", "/a"}, []string{"/a"}},
		{[]string{"-Wl,-rpath,/a", "-lfoo", "-Wl,-rpath=/a", "-Wl,-R,/a"}, []string{"/a"}},
		{[]string{"-Wl,-rpath,/a:/b"}, []string{"/a:/b"}},
		{[]string{"-Wl,-rpath-link,/a", "-Wl,-rpath-link=/b"}, nil},
		{[]string{"-Wl,-rpath", "-lfoo", "-Wl,/a"}, nil},
		{[]string{"-Wl,-rpath,"}, nil},
		{[]string{"-L/a", "-lfoo", "-Wl,--as-needed"}, nil},
	}
	defer func() { cgoRpaths = nil }()
	for _, tt := range tests {
		cgoRpaths = nil
		var p cgoRpathParser
		for _, f := range tt.flags {
			p.parse(f)
		}
		if !slices.Equal(cgoRpaths, tt.want) {
			t.Errorf("parse(%q): cgoRpaths = %q, want %q", tt.flags, cgoRpaths, tt.want)
		}
	}
}

func TestInternalRpaths(t *testing.T) {
	defer func(r Rpath, c []string) { rpath, cgoRpaths = r, c }(rpath, cgoRpaths)

	cgoRpaths = []string{"/a", "/b"}
	rpath = Rpath{}
	if got := internalRpaths(); !slices.Equal(got, []string{"/a", "/b"}) {
		t.Errorf("without -r: internalRpaths() = %q, want cgo rpaths", got)
	}
	rpath = Rpath{set: true, val: "/x:/y"}
	if got := internalRpaths(); !slices.Equal(got, []string{"/x", "/y"}) {
		t.Errorf("with -r /x:/y: internalRpaths() = %q, want [/x /y]", got)
	}
	rpath = Rpath{set: true}
	if got := internalRpaths(); len(got) != 0 {
		t.Errorf(`with -r "": internalRpaths() = %q, want none`, got)
	}
}

// TestMachoAddRpath verifies that the Mach-O header grows to hold the
// LC_RPATH load commands for many or long cgo rpath entries.
func TestMachoAddRpath(t *testing.T) {
	defer func(h int32, ta int64, b int, r Rpath, l []MachoLoad) {
		HEADR, *FlagTextAddr, loadBudget, rpath, load = h, ta, b, r, l
	}(HEADR, *FlagTextAddr, loadBudget, rpath, load)
	rpath = Rpath{}

	for _, tt := range []struct {
		name string
		dirs []string
	}{
		{"many", slices.Repeat([]string{"/" + strings.Repeat("d", 199)}, 100)},
		{"long", []string{"/" + strings.Repeat("d", 10000)}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			HEADR, *FlagTextAddr, loadBudget = INITIAL_MACHO_HEADR, 0, INITIAL_MACHO_HEADR-2*1024
			budget := loadBudget
			var need int
			for _, dir := range tt.dirs {
				machoaddrpath(dir, LinkInternal)
				// The size of the load command written by asmbMacho.
				need += 8 + 4*len(newMachoLoad(sys.ArchARM64, 0, uint32(1+(len(dir)+1+3)/4)).data)
			}
			load = nil
			grow := int(HEADR - INITIAL_MACHO_HEADR)
			if grow == 0 || grow%4096 != 0 || int64(grow) != *FlagTextAddr {
				t.Fatalf("HEADR grew by %d and -T by %d, want the same nonzero multiple of 4096", grow, *FlagTextAddr)
			}
			if loadBudget < 0 || budget+grow-need != loadBudget {
				t.Errorf("loadBudget = %d, want %d (budget %d + growth %d - load commands %d)", loadBudget, budget+grow-need, budget, grow, need)
			}
		})
	}

	// Nothing is reserved when linking externally or with -r.
	HEADR, loadBudget = INITIAL_MACHO_HEADR, 100
	machoaddrpath("/dir", LinkExternal)
	rpath = Rpath{set: true}
	machoaddrpath("/dir", LinkInternal)
	if HEADR != INITIAL_MACHO_HEADR || loadBudget != 100 {
		t.Errorf("HEADR = %d, loadBudget = %d; want unchanged", HEADR, loadBudget)
	}
}

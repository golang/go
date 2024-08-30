// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"cmd/internal/objabi"
	"cmd/internal/sys"
	"cmd/link/internal/loader"
	"internal/buildcfg"
	"testing"
)

func setUpContext(arch *sys.Arch, iself bool, ht objabi.HeadType, bm, lm string) *Link {
	ctxt := linknew(arch)
	ctxt.HeadType = ht
	er := loader.ErrorReporter{}
	ctxt.loader = loader.NewLoader(0, &er)
	ctxt.BuildMode.Set(bm)
	ctxt.LinkMode.Set(lm)
	ctxt.IsELF = iself
	ctxt.mustSetHeadType()
	ctxt.setArchSyms()
	return ctxt
}

// Make sure the addgotsym properly increases the symbols.
func TestAddGotSym(t *testing.T) {
	tests := []struct {
		arch    *sys.Arch
		ht      objabi.HeadType
		bm, lm  string
		rel     string
		relsize int
		gotsize int
	}{
		{
			arch:    sys.Arch386,
			ht:      objabi.Hlinux,
			bm:      "pie",
			lm:      "internal",
			rel:     ".rel",
			relsize: 2 * sys.Arch386.PtrSize,
			gotsize: sys.Arch386.PtrSize,
		},
		{
			arch:    sys.ArchAMD64,
			ht:      objabi.Hlinux,
			bm:      "pie",
			lm:      "internal",
			rel:     ".rela",
			relsize: 3 * sys.ArchAMD64.PtrSize,
			gotsize: sys.ArchAMD64.PtrSize,
		},
		{
			arch:    sys.ArchAMD64,
			ht:      objabi.Hdarwin,
			bm:      "pie",
			lm:      "external",
			gotsize: sys.ArchAMD64.PtrSize,
		},
	}

	// Save the architecture as we're going to set it on each test run.
	origArch := buildcfg.GOARCH
	defer func() {
		buildcfg.GOARCH = origArch
	}()

	for i, test := range tests {
		iself := len(test.rel) != 0
		buildcfg.GOARCH = test.arch.Name
		ctxt := setUpContext(test.arch, iself, test.ht, test.bm, test.lm)
		foo := ctxt.loader.CreateSymForUpdate("foo", 0)
		ctxt.loader.CreateExtSym("bar", 0)
		AddGotSym(&ctxt.Target, ctxt.loader, &ctxt.ArchSyms, foo.Sym(), 0)

		if iself {
			rel := ctxt.loader.Lookup(test.rel, 0)
			if rel == 0 {
				t.Fatalf("[%d] could not find symbol: %q", i, test.rel)
			}
			if s := ctxt.loader.SymSize(rel); s != int64(test.relsize) {
				t.Fatalf("[%d] expected ldr.Size(%q) == %v, got %v", i, test.rel, test.relsize, s)
			}
		}
		if s := ctxt.loader.SymSize(ctxt.loader.Lookup(".got", 0)); s != int64(test.gotsize) {
			t.Fatalf(`[%d] expected ldr.Size(".got") == %v, got %v`, i, test.gotsize, s)
		}
	}
}

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sve

import (
	"flag"
	"strings"
	"testing"
)

// svePath points at an extracted ARM64 ISA XML directory (see the download
// instructions in the package doc / TestRealXMLNoAnomalies). When unset, the
// corpus tests are skipped so the package still tests offline.
var svePath = flag.String("svePath", "", "path to extracted ISA_A64 XML directory")

// TestRealXMLNoAnomalies runs the loader across the whole SVE instruction set
// and fails if any instruction is classified as an anomaly — a form the loader
// does not understand. Deferred-but-recognized skips (memory, immediate, etc.)
// are allowed. This mirrors how instgen validates against the real XML.
//
// Download the data once:
//
//	curl -L -o isa.tgz https://developer.arm.com/-/cdn-downloads/permalink/Exploration-Tools-A64-ISA/ISA_A64/ISA_A64_xml_A_profile-2025-12.tar.gz
//	tar xzf isa.tgz
//
// then run:
//
//	go test ./simdgen/sve/ -run TestRealXML -svePath ./ISA_A64_xml_A_profile-2025-12 -v
func TestRealXMLNoAnomalies(t *testing.T) {
	if *svePath == "" {
		t.Skip("set -svePath to an extracted ISA_A64 XML directory")
	}
	r, err := analyze(*svePath)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("\n%s", r.String())
	if len(r.Anomalies) > 0 {
		t.Errorf("loader did not understand %d instruction form(s):\n  %s",
			len(r.Anomalies), strings.Join(r.Anomalies, "\n  "))
	}
}

// TestRealXMLAddFamily checks that the core Add-family instructions are emitted
// with the expected arrangements against the real XML.
func TestRealXMLAddFamily(t *testing.T) {
	if *svePath == "" {
		t.Skip("set -svePath to an extracted ISA_A64 XML directory")
	}
	insts, err := parseInstructions(*svePath)
	if err != nil {
		t.Fatal(err)
	}
	// mnemonic -> whether we saw it emit at least one def.
	want := map[string]bool{"ADD": false, "FADD": false, "SQADD": false, "UQADD": false}
	for _, inst := range insts {
		m := inst.mnemonic()
		if _, ok := want[m]; !ok {
			continue
		}
		if len(inst.emitAll()) > 0 {
			want[m] = true
		}
	}
	for m, ok := range want {
		if !ok {
			t.Errorf("expected %s to emit at least one def", m)
		}
	}
}

// TestRealXMLShiftImmediateWidths checks that a shift by an immediate, whose
// element size is encoded in tszh:tszl with wildcard rows, is emitted for
// every element width.
func TestRealXMLShiftImmediateWidths(t *testing.T) {
	if *svePath == "" {
		t.Skip("set -svePath to an extracted ISA_A64 XML directory")
	}
	defs, err := Load(*svePath)
	if err != nil {
		t.Fatal(err)
	}
	got := map[int]bool{}
	for _, d := range defs {
		var op struct {
			Asm string `unify:"asm"`
			In  []struct {
				Class    string
				ElemBits *int `unify:"elemBits"`
			} `unify:"in"`
			InVariant []struct{ Class string } `unify:"inVariant"`
		}
		if err := d.Decode(&op); err != nil {
			t.Fatal(err)
		}
		// The unpredicated form: an immediate and one vector, no predicate.
		if op.Asm != "ZLSL" || len(op.In) != 2 || op.In[0].Class != "immediate" || op.In[1].ElemBits == nil {
			continue
		}
		got[*op.In[1].ElemBits] = true
	}
	for _, w := range []int{8, 16, 32, 64} {
		if !got[w] {
			t.Errorf("LSL (immediate, unpredicated) has no %d-bit form; got %v", w, got)
		}
	}
}

// TestRealXMLImmediateDescriptions checks that a shift by an immediate carries
// the ISA's description of the immediate for its encoding, which simdgen
// resolves into a range.
func TestRealXMLImmediateDescriptions(t *testing.T) {
	if *svePath == "" {
		t.Skip("set -svePath to an extracted ISA_A64 XML directory")
	}
	defs, err := Load(*svePath)
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]string{
		"ZLSL": "Is the immediate shift amount, in the range 0 to number of bits per element minus 1, encoded in \"tszh:tszl:imm3\".",
		"ZASR": "Is the immediate shift amount, in the range 1 to number of bits per element, encoded in \"tszh:tszl:imm3\".",
	}
	seen := map[string]bool{}
	for _, d := range defs {
		var op struct {
			Asm string `unify:"asm"`
			In  []struct {
				Class   string
				ImmDesc *string `unify:"immDesc"`
			} `unify:"in"`
		}
		if err := d.Decode(&op); err != nil {
			t.Fatal(err)
		}
		w, ok := want[op.Asm]
		if !ok || len(op.In) != 2 || op.In[0].Class != "immediate" {
			continue
		}
		seen[op.Asm] = true
		if got := op.In[0].ImmDesc; got == nil || *got != w {
			t.Errorf("%s #imm: immDesc %v, want %q", op.Asm, got, w)
		}
	}
	for asm := range want {
		if !seen[asm] {
			t.Errorf("no unpredicated %s #imm def", asm)
		}
	}
}

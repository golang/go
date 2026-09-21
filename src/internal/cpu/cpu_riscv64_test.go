// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build riscv64

package cpu_test

import (
	. "internal/cpu"
	"internal/godebug"
	"testing"
)

func TestRISCV64VectorLength(t *testing.T) {
	if RISCV64.HasV && RISCV64.VLENB == 0 {
		t.Fatal("VLENB should be non-zero when HasV is true")
	}
}

func TestDisableZbb(t *testing.T) {
	if GetGORISCV64level() > 20 {
		t.Skip("skipping test: can't run on GORISCV64>rva20u64")
	}
	runDebugOptionsTest(t, "TestZbbDebugOption", "cpu.zbb=off")
}

func TestZbbDebugOption(t *testing.T) {
	MustHaveDebugOptionsSupport(t)

	if godebug.New("#cpu.zbb").Value() != "off" {
		t.Skipf("skipping test: GODEBUG=cpu.zbb=off not set")
	}

	if RISCV64.HasZbb {
		t.Errorf("RISCV64.HasZbb is true, want false")
	}
}

func TestDisableV(t *testing.T) {
	if GetGORISCV64level() > 22 {
		t.Skip("skipping test: can't run on GORISCV64>rva22u64")
	}

	runDebugOptionsTest(t, "TestVDebugOption", "cpu.v=off")
}

func TestVDebugOption(t *testing.T) {
	MustHaveDebugOptionsSupport(t)

	if godebug.New("#cpu.v").Value() != "off" {
		t.Skipf("skipping test: GODEBUG=cpu.v=off not set")
	}

	if RISCV64.HasV {
		t.Errorf("RISCV64.HasV is true, want false")
	}
}

func TestDisableZvbb(t *testing.T) {
	if GetGORISCV64level() > 22 {
		t.Skip("skipping test: can't run on GORISCV64>rva22u64")
	}

	runDebugOptionsTest(t, "TestZvbbDebugOption", "cpu.zvbb=off")
}

func TestZvbbDebugOption(t *testing.T) {
	MustHaveDebugOptionsSupport(t)

	if godebug.New("#cpu.zvbb").Value() != "off" {
		t.Skipf("skipping test: GODEBUG=cpu.zvbb=off not set")
	}

	if RISCV64.HasZvbb {
		t.Errorf("RISCV64.HasZvbb is true, want false")
	}
}

func TestRISCV64ZvbbHasV(t *testing.T) {
	if RISCV64.HasZvbb && !RISCV64.HasV {
		t.Fatalf("HasV is false when HasZvbb is true, want true")
	}
}

func TestRISCV64ZvbcHasV(t *testing.T) {
	if RISCV64.HasZvbc && !RISCV64.HasV {
		t.Fatalf("HasV is false when HasZvbc is true, want true")
	}
}

func TestRISCV64ZvkgHasV(t *testing.T) {
	if RISCV64.HasZvkg && !RISCV64.HasV {
		t.Fatalf("HasV is false when HasZvkg is true, want true")
	}
}

func TestRISCV64ZvknedHasV(t *testing.T) {
	if RISCV64.HasZvkned && !RISCV64.HasV {
		t.Fatalf("HasV is false when HasZvkned is true, want true")
	}
}

func TestRISCV64ZvknhaHasV(t *testing.T) {
	if RISCV64.HasZvknha && !RISCV64.HasV {
		t.Fatalf("HasV is false when HasZvknha is true, want true")
	}
}

func TestRISCV64ZvknhbHasV(t *testing.T) {
	if RISCV64.HasZvknhb && !RISCV64.HasV {
		t.Fatalf("HasV is false when HasZvknhb is true, want true")
	}
}

func TestRISCV64ZvksedHasV(t *testing.T) {
	if RISCV64.HasZvksed && !RISCV64.HasV {
		t.Fatalf("HasV is false when HasZvksed is true, want true")
	}
}

func TestRISCV64ZvkshHasV(t *testing.T) {
	if RISCV64.HasZvksh && !RISCV64.HasV {
		t.Fatalf("HasV is false when HasZvksh is true, want true")
	}
}

func TestRISCV64ZvktHasV(t *testing.T) {
	if RISCV64.HasZvkt && !RISCV64.HasV {
		t.Fatalf("HasV is false when HasZvkt is true, want true")
	}
}

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fipstest

import (
	. "crypto/internal/fips140/check"
	"crypto/internal/fips140/check/checktest"
	"fmt"
	"internal/abi"
	"internal/asan"
	"internal/godebug"
	"internal/testenv"
	"os"
	"runtime"
	"testing"
	"unicode"
	"unsafe"
)

const enableFIPSTest = true

func TestFIPSCheckVerify(t *testing.T) {
	if Verified {
		t.Logf("verified")
		return
	}

	if godebug.New("#fips140").Value() == "on" {
		t.Fatalf("GODEBUG=fips140=on but verification did not run")
	}

	if !enableFIPSTest {
		return
	}

	if !Supported() {
		t.Skipf("skipping on %s-%s", runtime.GOOS, runtime.GOARCH)
	}
	if asan.Enabled {
		// Verification panics with asan; don't bother.
		t.Skipf("skipping with -asan")
	}

	cmd := testenv.Command(t, os.Args[0], "-test.v", "-test.run=TestFIPSCheck")
	cmd.Env = append(cmd.Environ(), "GODEBUG=fips140=on")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("GODEBUG=fips140=on %v failed: %v\n%s", cmd.Args, err, out)
	}
	t.Logf("exec'ed GODEBUG=fips140=on and succeeded:\n%s", out)
}

func TestFIPSCheckInfo(t *testing.T) {
	if !enableFIPSTest {
		return
	}

	if !Supported() {
		t.Skipf("skipping on %s-%s", runtime.GOOS, runtime.GOARCH)
	}

	// Check that the checktest symbols are initialized properly.
	if checktest.NOPTRDATA != 1 {
		t.Errorf("checktest.NOPTRDATA = %d, want 1", checktest.NOPTRDATA)
	}
	if checktest.RODATA != 2 {
		t.Errorf("checktest.RODATA = %d, want 2", checktest.RODATA)
	}
	if checktest.DATA.P != &checktest.NOPTRDATA {
		t.Errorf("checktest.DATA.P = %p, want &checktest.NOPTRDATA (%p)", checktest.DATA.P, &checktest.NOPTRDATA)
	}
	if checktest.DATA.X != 3 {
		t.Errorf("checktest.DATA.X = %d, want 3", checktest.DATA.X)
	}
	if checktest.NOPTRBSS != 0 {
		t.Errorf("checktest.NOPTRBSS = %d, want 0", checktest.NOPTRBSS)
	}
	if checktest.BSS != nil {
		t.Errorf("checktest.BSS = %p, want nil", checktest.BSS)
	}
	if p := checktest.PtrStaticData(); p != nil && *p != 10 {
		t.Errorf("*checktest.PtrStaticData() = %d, want 10", *p)
	}

	// Check that the checktest symbols are in the right go:fipsinfo sections.
	sect := func(i int, name string, p unsafe.Pointer) {
		s := Linkinfo.Sects[i]
		if !(uintptr(s.Start) <= uintptr(p) && uintptr(p) < uintptr(s.End)) {
			t.Errorf("checktest.%s (%#x) not in section #%d (%#x..%#x)", name, p, i, s.Start, s.End)
		}
	}
	sect(0, "TEXT", unsafe.Pointer(abi.FuncPCABIInternal(checktest.TEXT)))
	if p := checktest.PtrStaticText(); p != nil {
		sect(0, "StaticText", p)
	}
	sect(1, "RODATA", unsafe.Pointer(&checktest.RODATA))
	sect(2, "NOPTRDATA", unsafe.Pointer(&checktest.NOPTRDATA))
	if p := checktest.PtrStaticData(); p != nil {
		sect(2, "StaticData", unsafe.Pointer(p))
	}
	sect(3, "DATA", unsafe.Pointer(&checktest.DATA))

	// Check that some symbols are not in FIPS sections.
	no := func(name string, p unsafe.Pointer, ix ...int) {
		for _, i := range ix {
			s := Linkinfo.Sects[i]
			if uintptr(s.Start) <= uintptr(p) && uintptr(p) < uintptr(s.End) {
				t.Errorf("%s (%#x) unexpectedly in section #%d (%#x..%#x)", name, p, i, s.Start, s.End)
			}
		}
	}

	// Check that the symbols are not in unexpected sections (that is, no overlaps).
	no("checktest.TEXT", unsafe.Pointer(abi.FuncPCABIInternal(checktest.TEXT)), 1, 2, 3)
	no("checktest.RODATA", unsafe.Pointer(&checktest.RODATA), 0, 2, 3)
	no("checktest.NOPTRDATA", unsafe.Pointer(&checktest.NOPTRDATA), 0, 1, 3)
	no("checktest.DATA", unsafe.Pointer(&checktest.DATA), 0, 1, 2)

	// Check that non-FIPS symbols are not in any of the sections.
	no("fmt.Printf", unsafe.Pointer(abi.FuncPCABIInternal(fmt.Printf)), 0, 1, 2, 3)     // TEXT
	no("unicode.Categories", unsafe.Pointer(&unicode.Categories), 0, 1, 2, 3)           // BSS
	no("unicode.ASCII_Hex_Digit", unsafe.Pointer(&unicode.ASCII_Hex_Digit), 0, 1, 2, 3) // DATA

	// Check that we have enough data in total.
	// On arm64 the fips sections in this test currently total 23 kB.
	n := uintptr(0)
	for _, s := range Linkinfo.Sects {
		n += uintptr(s.End) - uintptr(s.Start)
	}
	if n < 16*1024 {
		t.Fatalf("fips sections not big enough: %d, want at least 16 kB", n)
	}
}

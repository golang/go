// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fipstest

import (
	"bytes"
	"crypto/internal/cryptotest"
	. "crypto/internal/fips140/check"
	"crypto/internal/fips140/check/checktest"
	"fmt"
	"internal/abi"
	"internal/godebug"
	"internal/testenv"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
	"unicode"
	"unsafe"
)

func TestIntegrityCheck(t *testing.T) {
	if Verified {
		t.Logf("verified")
		return
	}

	if godebug.New("fips140").Value() == "on" {
		t.Fatalf("GODEBUG=fips140=on but verification did not run")
	}

	cryptotest.RerunWithFIPS140Enabled(t)
}

func TestIntegrityCheckFailure(t *testing.T) {
	moduleStatus(t)
	cryptotest.MustSupportFIPS140(t)

	t.Logf("running modified binary...")
	cmd := reexecCommand(t, "fips140=on", true, "-test.v", "-test.run=^TestIntegrityCheck$")
	out, err := cmd.CombinedOutput()
	t.Logf("running with GODEBUG=fips140=on:\n%s", out)
	if err == nil {
		t.Errorf("modified binary did not fail as expected")
	}
	if !bytes.Contains(out, []byte("fips140: verification mismatch")) {
		t.Errorf("modified binary did not fail with expected message")
	}
	if bytes.Contains(out, []byte("verified")) {
		t.Errorf("modified binary did not exit")
	}
}

// browserBridge is the path to the browserbridge client binary, to be used to re-exec
// self-tests. This lets the tests run on the host while exercising a js/wasm
// module, which can't exec a subprocess. See crypto/internal/fips140test/_browserbridge.
var browserBridge = os.Getenv("GOBROWSERBRIDGE")

func reexecCommand(t *testing.T, godebug string, corrupt bool, args ...string) *exec.Cmd {
	if browserBridge == "" {
		exe := testenv.Executable(t)
		if corrupt {
			exe = corruptExecutable(t)
		}
		cmd := testenv.Command(t, exe, args...)
		cmd.Env = append(cmd.Environ(), "GODEBUG="+godebug)
		return cmd
	}

	bridgeArgs := []string{"-run"}
	if corrupt {
		bridgeArgs = append(bridgeArgs, "-corrupt")
	}
	bridgeArgs = append(append(bridgeArgs, "--"), args...)
	cmd := testenv.Command(t, browserBridge, bridgeArgs...)
	cmd.Env = append(cmd.Environ(), "GODEBUG="+godebug)
	return cmd
}

func corruptExecutable(t *testing.T) string {
	bin, err := os.ReadFile(testenv.Executable(t))
	if err != nil {
		t.Fatal(err)
	}

	// Replace the expected module checksum with a different value.
	bin = bytes.ReplaceAll(bin, Linkinfo.Sum[:], bytes.Repeat([]byte("X"), len(Linkinfo.Sum)))

	binPath := filepath.Join(t.TempDir(), "fips140test.exe")
	if err := os.WriteFile(binPath, bin, 0o755); err != nil {
		t.Fatal(err)
	}

	if runtime.GOOS == "darwin" {
		// Regenerate the macOS ad-hoc code signature.
		cmd := testenv.Command(t, "codesign", "-s", "-", "-f", binPath)
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("codesign failed: %v\n%s", err, out)
		}
	}

	return binPath
}

func TestIntegrityCheckInfo(t *testing.T) {
	cryptotest.MustSupportFIPS140(t)

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

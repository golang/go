// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"debug/elf"
	"debug/macho"
	"encoding/binary"
	"internal/testenv"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func TestCgoUnwind(t *testing.T) {
	// Test that the linker emits static unwind info for runtime/cgo trampolines
	// in runtimecgopurego binaries and omits it from pure-Go binaries.
	if testing.Short() &&
		(runtime.GOOS != "linux" && runtime.GOOS != "darwin" ||
			runtime.GOARCH != "amd64" && runtime.GOARCH != "arm64") {
		t.Skipf("skipping cross-compile test on %s/%s in short mode", runtime.GOOS, runtime.GOARCH)
	}
	testenv.MustHaveGoBuild(t)
	t.Parallel()
	tmpdir := t.TempDir()

	cgoSrc := filepath.Join(tmpdir, "cgo.go")
	if err := os.WriteFile(cgoSrc, []byte("package main\nimport _ \"runtime/cgo\"\nfunc main() {}\n"), 0644); err != nil {
		t.Fatal(err)
	}
	pureSrc := filepath.Join(tmpdir, "pure.go")
	if err := os.WriteFile(pureSrc, []byte("package main\nfunc main() {}\n"), 0644); err != nil {
		t.Fatal(err)
	}

	build := func(t *testing.T, goos, goarch, cgoEnabled, src string, extra ...string) string {
		t.Helper()
		outPath := filepath.Join(tmpdir, t.Name()+".out")
		if err := os.MkdirAll(filepath.Dir(outPath), 0755); err != nil {
			t.Fatal(err)
		}
		args := append([]string{"build", "-ldflags=-linkmode=internal", "-o", outPath}, extra...)
		args = append(args, src)
		cmd := testenv.Command(t, testenv.GoToolPath(t), args...)
		cmd.Env = append(os.Environ(),
			"GOOS="+goos,
			"GOARCH="+goarch,
			"CGO_ENABLED="+cgoEnabled,
			"GOEXPERIMENT=runtimecgopurego",
		)
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("go build failed: %v\n%s", err, out)
		}
		return outPath
	}

	elfTests := []struct {
		name         string
		goos, goarch string
		extra        []string
	}{
		{"linux_amd64", "linux", "amd64", nil},
		{"linux_amd64_pie", "linux", "amd64", []string{"-buildmode=pie"}},
		{"linux_arm64", "linux", "arm64", nil},
		{"linux_arm64_pie", "linux", "arm64", []string{"-buildmode=pie"}},
	}
	for _, tt := range elfTests {
		if testing.Short() && (tt.goos != runtime.GOOS || tt.goarch != runtime.GOARCH) {
			continue
		}
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			exe := build(t, tt.goos, tt.goarch, "1", cgoSrc, tt.extra...)
			verifyELFCgoUnwind(t, exe)
		})
	}

	darwinTests := []struct {
		name     string
		goarch   string
		encoding uint32
	}{
		{"darwin_amd64", "amd64", 0x01000000},
		{"darwin_arm64", "arm64", 0x04000000},
	}
	for _, tt := range darwinTests {
		if testing.Short() && (runtime.GOOS != "darwin" || tt.goarch != runtime.GOARCH) {
			continue
		}
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			exe := build(t, "darwin", tt.goarch, "1", cgoSrc)
			verifyMachoCgoUnwind(t, exe, tt.encoding)
		})
	}

	if !testing.Short() || (runtime.GOOS == "linux" && runtime.GOARCH == "amd64") {
		t.Run("pure_linux_amd64", func(t *testing.T) {
			t.Parallel()
			exe := build(t, "linux", "amd64", "0", pureSrc)
			f, err := elf.Open(exe)
			if err != nil {
				t.Fatal(err)
			}
			defer f.Close()
			if f.Section(".eh_frame") != nil || f.Section(".eh_frame_hdr") != nil {
				t.Fatalf("unexpected .eh_frame section in pure-Go binary")
			}
			for _, p := range f.Progs {
				if p.Type == elf.PT_GNU_EH_FRAME {
					t.Fatalf("unexpected PT_GNU_EH_FRAME in pure-Go binary")
				}
			}
		})
	}

	if !testing.Short() || (runtime.GOOS == "darwin" && runtime.GOARCH == "arm64") {
		t.Run("pure_darwin_arm64", func(t *testing.T) {
			t.Parallel()
			exe := build(t, "darwin", "arm64", "0", pureSrc)
			f, err := macho.Open(exe)
			if err != nil {
				t.Fatal(err)
			}
			defer f.Close()
			if f.Section("__unwind_info") != nil {
				t.Fatalf("unexpected __unwind_info section in pure-Go binary")
			}
		})
	}
}

func verifyELFCgoUnwind(t *testing.T, path string) {
	t.Helper()
	f, err := elf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	eh := f.Section(".eh_frame")
	hdr := f.Section(".eh_frame_hdr")
	if eh == nil || hdr == nil {
		t.Fatalf("missing .eh_frame (%v) or .eh_frame_hdr (%v)", eh != nil, hdr != nil)
	}
	ehData, err := eh.Data()
	if err != nil {
		t.Fatal(err)
	}
	hdrData, err := hdr.Data()
	if err != nil {
		t.Fatal(err)
	}
	if len(ehData) < 28 || binary.LittleEndian.Uint32(ehData[len(ehData)-4:]) != 0 {
		t.Fatalf(".eh_frame missing trailing zero terminator")
	}

	var foundPhdr bool
	for _, p := range f.Progs {
		if p.Type == elf.PT_GNU_EH_FRAME {
			foundPhdr = true
			if p.Vaddr != hdr.Addr || p.Off != hdr.Offset || p.Filesz != hdr.Size {
				t.Fatalf("PT_GNU_EH_FRAME (%#x,%#x,%d) != .eh_frame_hdr (%#x,%#x,%d)",
					p.Vaddr, p.Off, p.Filesz, hdr.Addr, hdr.Offset, hdr.Size)
			}
		}
	}
	if !foundPhdr {
		t.Fatalf("missing PT_GNU_EH_FRAME program header")
	}

	if hdrData[0] != 1 || hdrData[1] != 0x1b || hdrData[2] != 0x03 || hdrData[3] != 0x3b {
		t.Fatalf("unexpected .eh_frame_hdr header: %x", hdrData[:4])
	}
	ehPtr := uint64(int64(hdr.Addr+4) + int64(int32(binary.LittleEndian.Uint32(hdrData[4:8]))))
	if ehPtr != eh.Addr {
		t.Fatalf("eh_frame_ptr = %#x, want %#x", ehPtr, eh.Addr)
	}
	fdeCount := binary.LittleEndian.Uint32(hdrData[8:12])
	if int(fdeCount) != len(cgoUnwindSyms) {
		t.Fatalf("fde_count = %d, want %d", fdeCount, len(cgoUnwindSyms))
	}

	syms, err := f.Symbols()
	if err != nil {
		t.Fatal(err)
	}
	wantSyms := map[uint64]uint64{}
	for _, s := range syms {
		for _, name := range cgoUnwindSyms {
			if s.Name == name {
				wantSyms[s.Value] = s.Size
			}
		}
	}
	if len(wantSyms) != len(cgoUnwindSyms) {
		t.Fatalf("found %d cgo unwind symbols, want %d", len(wantSyms), len(cgoUnwindSyms))
	}

	var prevLoc uint64
	for i := uint32(0); i < fdeCount; i++ {
		off := 12 + i*8
		loc := uint64(int64(hdr.Addr) + int64(int32(binary.LittleEndian.Uint32(hdrData[off:off+4]))))
		fdeAddr := uint64(int64(hdr.Addr) + int64(int32(binary.LittleEndian.Uint32(hdrData[off+4:off+8]))))
		if i > 0 && loc <= prevLoc {
			t.Fatalf("search table not sorted: %#x <= %#x", loc, prevLoc)
		}
		prevLoc = loc

		fdeOff := fdeAddr - eh.Addr
		ciePtr := binary.LittleEndian.Uint32(ehData[fdeOff+4 : fdeOff+8])
		initLoc := uint64(int64(fdeAddr+8) + int64(int32(binary.LittleEndian.Uint32(ehData[fdeOff+8:fdeOff+12]))))
		addrRange := uint64(binary.LittleEndian.Uint32(ehData[fdeOff+12 : fdeOff+16]))
		if uint64(ciePtr) != fdeOff+4 {
			t.Fatalf("FDE %d CIE_pointer = %#x, want %#x", i, ciePtr, fdeOff+4)
		}
		if initLoc != loc {
			t.Fatalf("FDE %d initial_location = %#x, want %#x", i, initLoc, loc)
		}
		wantSize, ok := wantSyms[loc]
		if !ok {
			t.Fatalf("FDE %d initial_location %#x does not match any cgo unwind symbol", i, loc)
		}
		if addrRange != wantSize {
			t.Fatalf("FDE %d address_range = %d, want %d", i, addrRange, wantSize)
		}
	}
}

func verifyMachoCgoUnwind(t *testing.T, path string, wantEnc uint32) {
	t.Helper()
	f, err := macho.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	sec := f.Section("__unwind_info")
	if sec == nil || sec.Seg != "__TEXT" {
		t.Fatalf("missing __TEXT,__unwind_info section")
	}
	data, err := sec.Data()
	if err != nil {
		t.Fatal(err)
	}
	textSeg := f.Segment("__TEXT")
	if textSeg == nil {
		t.Fatalf("missing __TEXT segment")
	}

	if binary.LittleEndian.Uint32(data[0:4]) != 1 {
		t.Fatalf("unexpected __unwind_info version %d", binary.LittleEndian.Uint32(data[0:4]))
	}
	indexOff := binary.LittleEndian.Uint32(data[20:24])
	indexCount := binary.LittleEndian.Uint32(data[24:28])
	if indexCount != 2 {
		t.Fatalf("indexCount = %d, want 2", indexCount)
	}
	pageOff := binary.LittleEndian.Uint32(data[indexOff+4 : indexOff+8])
	if binary.LittleEndian.Uint32(data[pageOff:pageOff+4]) != 2 {
		t.Fatalf("unexpected second-level page kind %d", binary.LittleEndian.Uint32(data[pageOff:pageOff+4]))
	}
	entOff := pageOff + uint32(binary.LittleEndian.Uint16(data[pageOff+4:pageOff+6]))
	entCount := uint32(binary.LittleEndian.Uint16(data[pageOff+6 : pageOff+8]))

	gotEnc := map[uint32]uint32{}
	for i := uint32(0); i < entCount; i++ {
		off := entOff + i*8
		fnOff := binary.LittleEndian.Uint32(data[off : off+4])
		enc := binary.LittleEndian.Uint32(data[off+4 : off+8])
		gotEnc[fnOff] = enc
	}

	if f.Symtab == nil {
		t.Fatalf("missing Mach-O symtab")
	}
	matched := 0
	for _, s := range f.Symtab.Syms {
		for _, name := range cgoUnwindSyms {
			if s.Name == name {
				fnOff := uint32(s.Value - textSeg.Addr)
				if gotEnc[fnOff] != wantEnc {
					t.Fatalf("%s (off %#x): encoding = %#x, want %#x", name, fnOff, gotEnc[fnOff], wantEnc)
				}
				matched++
			}
		}
	}
	if matched != len(cgoUnwindSyms) {
		t.Fatalf("matched %d cgo unwind symbols, want %d", matched, len(cgoUnwindSyms))
	}
}

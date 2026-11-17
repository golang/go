// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testlogos

import (
	"internal/asan"
	"internal/goexperiment"
	"internal/msan"
	"internal/race"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestCgoOptPointerFreeAddress(t *testing.T) {
	requireCgo(t)
	experiment := "nologoscgoopt,nologosfastcgo"
	if goexperiment.LogosCgoOpt {
		experiment = "logoscgoopt,nologosfastcgo"
	}
	if goexperiment.CgoCheck2 {
		experiment += ",cgocheck2"
	}
	t.Setenv("GOEXPERIMENT", experiment)
	t.Setenv("GODEBUG", "cgocheck=1")
	dir, err := filepath.Abs(filepath.Join("testdata", "cgoaddr"))
	if err != nil {
		t.Fatal(err)
	}
	args := []string{"run"}
	switch {
	case race.Enabled:
		args = append(args, "-race")
	case asan.Enabled:
		args = append(args, "-asan")
	case msan.Enabled:
		args = append(args, "-msan")
	}
	for _, tool := range []string{goBin(t), logosBin(t)} {
		out, err := runCmd(t, dir, tool, append(args, ".")...)
		if err != nil {
			t.Fatalf("typed-address fixture: %v\n%s", err, out)
		}
		if strings.TrimSpace(out) != "typed-address, pinning, and deferred-evaluation checks ok" {
			t.Fatalf("unexpected typed-address fixture output: %q", out)
		}
	}
}

// The small eligibility predicate must not add an out-of-line runtime call.
func TestCgoOptPointerFreeAddressInline(t *testing.T) {
	body := objdumpFunc(t, selfTestBin(t), "^runtime\\.cgoCheckPointer$")
	if !strings.Contains(body, "TEXT runtime.cgoCheckPointer") {
		t.Fatalf("missing cgoCheckPointer disassembly:\n%s", body)
	}
	if strings.Contains(body, "CALL runtime.logosCgoPointerFreeAddr(SB)") {
		t.Fatalf("pointer-free address predicate did not inline:\n%s", body)
	}
}

func BenchmarkCgoOptPointerFreeAddress(b *testing.B) {
	storage := new([32]int)
	packet := &cgoOptAddrPacket{header: [16]byte{17}, next: &storage[0]}
	b.ReportAllocs()
	for b.Loop() {
		cgoOptConsumeAddr(packet)
	}
	if got := cgoOptAddrObserved(); got != 17 {
		b.Fatalf("C observed %d, want 17", got)
	}
	runtime.KeepAlive(packet)
}

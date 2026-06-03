// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cpu_test

import (
	"internal/cpu"
	"internal/syscall/windows"
	"runtime"
	"testing"
)

func TestARM64WindowsFeatures(t *testing.T) {
	if runtime.GOARCH != "amd64" {
		return
	}

	if cpu.ARM64.HasCPUID {
		t.Fatal("HasCPUID expected false, got true")
	}
	if cpu.ARM64.HasDIT {
		t.Fatal("HasDIT expected false, got true")
	}
	if cpu.ARM64.IsNeoverse {
		t.Fatal("IsNeoverse expected false, got true")
	}

	if got, want := cpu.ARM64.HasAES, windows.IsProcessorFeaturePresent(windows.PF_ARM_V8_CRYPTO_INSTRUCTIONS_AVAILABLE); got != want {
		t.Fatalf("HasAES expected %v, got %v", want, got)
	}
	if got, want := cpu.ARM64.HasPMULL, windows.IsProcessorFeaturePresent(windows.PF_ARM_V8_CRYPTO_INSTRUCTIONS_AVAILABLE); got != want {
		t.Fatalf("HasPMULL expected %v, got %v", want, got)
	}
	if got, want := cpu.ARM64.HasSHA1, windows.IsProcessorFeaturePresent(windows.PF_ARM_V8_CRYPTO_INSTRUCTIONS_AVAILABLE); got != want {
		t.Fatalf("HasSHA1 expected %v, got %v", want, got)
	}
	if got, want := cpu.ARM64.HasSHA2, windows.IsProcessorFeaturePresent(windows.PF_ARM_V8_CRYPTO_INSTRUCTIONS_AVAILABLE); got != want {
		t.Fatalf("HasSHA2 expected %v, got %v", want, got)
	}

	if got, want := cpu.ARM64.HasSHA3, windows.IsProcessorFeaturePresent(windows.PF_ARM_SHA3_INSTRUCTIONS_AVAILABLE); got != want {
		t.Fatalf("HasSHA3 expected %v, got %v", want, got)
	}
	if got, want := cpu.ARM64.HasCRC32, windows.IsProcessorFeaturePresent(windows.PF_ARM_V8_CRC32_INSTRUCTIONS_AVAILABLE); got != want {
		t.Fatalf("HasCRC32 expected %v, got %v", want, got)
	}
	if got, want := cpu.ARM64.HasSHA512, windows.IsProcessorFeaturePresent(windows.PF_ARM_SHA512_INSTRUCTIONS_AVAILABLE); got != want {
		t.Fatalf("HasSHA512 expected %v, got %v", want, got)
	}
	if got, want := cpu.ARM64.HasATOMICS, windows.IsProcessorFeaturePresent(windows.PF_ARM_V81_ATOMIC_INSTRUCTIONS_AVAILABLE); got != want {
		t.Fatalf("HasATOMICS expected %v, got %v", want, got)
	}
}

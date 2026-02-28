// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sys

import (
	"testing"
)

func TestArchInFamily(t *testing.T) {
	if got, want := ArchPPC64LE.InFamily(AMD64), false; got != want {
		t.Errorf("Got ArchPPC64LE.InFamily(AMD64) = %v, want %v", got, want)
	}
	if got, want := ArchPPC64LE.InFamily(PPC64), true; got != want {
		t.Errorf("Got ArchPPC64LE.InFamily(PPC64) = %v, want %v", got, want)
	}
	if got, want := ArchPPC64LE.InFamily(AMD64, RISCV64), false; got != want {
		t.Errorf("Got ArchPPC64LE.InFamily(AMD64, RISCV64) = %v, want %v", got, want)
	}
	if got, want := ArchPPC64LE.InFamily(AMD64, PPC64), true; got != want {
		t.Errorf("Got ArchPPC64LE.InFamily(AMD64, PPC64) = %v, want %v", got, want)
	}
}

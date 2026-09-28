// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"fmt"
	. "runtime"
	"testing"
)

// TestMemmoveRealign runs long copies through both realignment strategies,
// aligned loads and aligned stores, whichever one the host CPU uses.
func TestMemmoveRealign(t *testing.T) {
	sizes := []int{129, 143, 144, 160, 255, 511, 1000, 1023, 1024, 1025, 2049}
	old := *Arm64UseAlignedLoads
	defer func() { *Arm64UseAlignedLoads = old }()
	for _, aligned := range []bool{false, true} {
		t.Run(fmt.Sprintf("alignedLoads=%v", aligned), func(t *testing.T) {
			*Arm64UseAlignedLoads = aligned
			for _, n := range sizes {
				buf := make([]byte, 2*n+64)
				want := make([]byte, len(buf))
				for so := 0; so < 16; so++ {
					for do := 0; do < 16; do++ {
						cases := []struct{ src, dst int }{
							{so, n + 32 + do},  // disjoint
							{32 + so, 49 + do}, // overlapping, dst after src
							{32 + so, 15 + do}, // overlapping, dst before src
						}
						for _, c := range cases {
							for i := range buf {
								buf[i] = byte(i * 7)
							}
							copy(want, buf)
							for i := 0; i < n; i++ {
								want[c.dst+i] = buf[c.src+i]
							}
							copy(buf[c.dst:c.dst+n], buf[c.src:c.src+n])
							for i := range buf {
								if buf[i] != want[i] {
									t.Fatalf("n=%d src=%d dst=%d: byte %d = %d, want %d", n, c.src, c.dst, i, buf[i], want[i])
								}
							}
						}
					}
				}
			}
		})
	}
}

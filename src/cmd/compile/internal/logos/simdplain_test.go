// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"strings"
	"testing"
)

func TestSimdSlicePlainLoopHasNoMaskedTail(t *testing.T) {
	for _, arch := range []string{"amd64", "arm64"} {
		t.Run(arch, func(t *testing.T) {
			simdSliceTarget(t, arch, 1)
			out := lowerSliceFunc(t, "xs []int, ys []int32", "int", "n := 0\n\tfor _, y := range ys {\n\t\tn += int(y)\n\t}\n\treturn n + xs.sum()")
			for _, name := range []string{"SliceMasked", "_logos_simd", "archsimd"} {
				if strings.Contains(out, name) {
					t.Errorf("non-reduction code names %s:\n%s", name, out)
				}
			}
			if !strings.Contains(out, "case 0: return 0") {
				t.Errorf("scalar sum loses its empty identity:\n%s", out)
			}
		})
	}
}

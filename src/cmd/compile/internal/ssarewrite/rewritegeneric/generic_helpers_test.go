// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rewritegeneric

import "testing"

func TestModularMultiplicativeInverse(t *testing.T) {
	t.Parallel()

	// We've got 63 bits of phase space for the Multiplier
	// Needless to say this is too much to bruteforce here.
	// I've randomly picked a range of 1<<24 because it runs in 0.03s on my machine which isn't too slow.
	// We test both sides of the wrapping point (0 and math.MaxUint64) since we need to test something and it's a usual place to have bugs.
	const halfRange = 1 << 23
	for i := -int64(halfRange) - 1; i < halfRange; i += 2 { // odd only, a bit after to a bit before the wrapping point
		mmi := modularMultiplicativeInverse(uint64(i))

		if uint64(i)*mmi != 1 {
			t.Errorf("%d * modularMultiplicativeInverse(%d) != 1; modularMultiplicativeInverse(%d) == %d", i, i, i, mmi)
		}
	}
}

// compile -t 30

//go:build !loong64

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Currently, this test causes a timeout when running as
// part of cmd/internal/testdir on loong64, even though
// the program itself is not slow when executed in isolation.

// TODO: after the root cause of the timeout is determined, the
// test should be re-enabled for loong64.

package p

import (
	"math/bits"
)

func ExpandKey(keys []uint64) []uint64 {
	w := make([]uint64, 72)

	w[0] = keys[0]
	w[1] = keys[1]
	w[2] = keys[2]
	w[3] = keys[3]
	w[4] = ks256(w[3], w[0], w[1], 1)
	w[5] = ks256(w[4], w[1], w[2], 1)
	w[6] = ks256(w[5], w[2], w[3], 0)
	w[7] = ks256(w[6], w[3], w[4], 1)
	w[8] = ks256(w[7], w[4], w[5], 0)
	w[9] = ks256(w[8], w[5], w[6], 0)
	w[10] = ks256(w[9], w[6], w[7], 0)
	w[11] = ks256(w[10], w[7], w[8], 1)
	w[12] = ks256(w[11], w[8], w[9], 1)
	w[13] = ks256(w[12], w[9], w[10], 1)
	w[14] = ks256(w[13], w[10], w[11], 1)
	w[15] = ks256(w[14], w[11], w[12], 0)
	w[16] = ks256(w[15], w[12], w[13], 0)
	w[17] = ks256(w[16], w[13], w[14], 1)
	w[18] = ks256(w[17], w[14], w[15], 1)
	w[19] = ks256(w[18], w[15], w[16], 0)
	w[20] = ks256(w[19], w[16], w[17], 1)
	w[21] = ks256(w[20], w[17], w[18], 0)
	w[22] = ks256(w[21], w[18], w[19], 1)
	w[23] = ks256(w[22], w[19], w[20], 1)
	w[24] = ks256(w[23], w[20], w[21], 0)
	w[25] = ks256(w[24], w[21], w[22], 1)
	w[26] = ks256(w[25], w[22], w[23], 1)
	w[27] = ks256(w[26], w[23], w[24], 0)
	w[28] = ks256(w[27], w[24], w[25], 0)
	w[29] = ks256(w[28], w[25], w[26], 0)
	w[30] = ks256(w[29], w[26], w[27], 1)
	w[31] = ks256(w[30], w[27], w[28], 0)
	w[32] = ks256(w[31], w[28], w[29], 0)
	w[33] = ks256(w[32], w[29], w[30], 0)
	w[34] = ks256(w[33], w[30], w[31], 0)
	w[35] = ks256(w[34], w[31], w[32], 0)
	w[36] = ks256(w[35], w[32], w[33], 0)
	w[37] = ks256(w[36], w[33], w[34], 1)
	w[38] = ks256(w[37], w[34], w[35], 0)
	w[39] = ks256(w[38], w[35], w[36], 1)
	w[40] = ks256(w[39], w[36], w[37], 1)
	w[41] = ks256(w[40], w[37], w[38], 1)
	w[42] = ks256(w[41], w[38], w[39], 0)
	w[43] = ks256(w[42], w[39], w[40], 0)
	w[44] = ks256(w[43], w[40], w[41], 0)
	w[45] = ks256(w[44], w[41], w[42], 0)
	w[46] = ks256(w[45], w[42], w[43], 1)
	w[47] = ks256(w[46], w[43], w[44], 1)
	w[48] = ks256(w[47], w[44], w[45], 0)
	w[49] = ks256(w[48], w[45], w[46], 0)
	w[50] = ks256(w[49], w[46], w[47], 1)
	w[51] = ks256(w[50], w[47], w[48], 0)
	w[52] = ks256(w[51], w[48], w[49], 1)
	w[53] = ks256(w[52], w[49], w[50], 0)
	w[54] = ks256(w[53], w[50], w[51], 0)
	w[55] = ks256(w[54], w[51], w[52], 1)
	w[56] = ks256(w[55], w[52], w[53], 0)
	w[57] = ks256(w[56], w[53], w[54], 0)
	w[58] = ks256(w[57], w[54], w[55], 1)
	w[59] = ks256(w[58], w[55], w[56], 1)
	w[60] = ks256(w[59], w[56], w[57], 1)
	w[61] = ks256(w[60], w[57], w[58], 0)
	w[62] = ks256(w[61], w[58], w[59], 1)
	w[63] = ks256(w[62], w[59], w[60], 1)
	w[64] = ks256(w[63], w[60], w[61], 1)
	w[65] = ks256(w[64], w[61], w[62], 1)
	w[66] = ks256(w[65], w[62], w[63], 1)
	w[67] = ks256(w[66], w[63], w[64], 1)
	w[68] = ks256(w[67], w[64], w[65], 0)
	w[69] = ks256(w[68], w[65], w[66], 1)
	w[70] = ks256(w[69], w[66], w[67], 0)
	w[71] = ks256(w[70], w[67], w[68], 0)

	return w
}

func rotatel64(x uint64, n int) uint64 {
	return bits.RotateLeft64(x, n)
}

func rotater64(x uint64, n int) uint64 {
	return rotatel64(x, 64-n)
}

func ks256(iplus3, i, iplus1 uint64, z int) uint64 {
	tmp := rotater64(iplus3, 3) ^ iplus1
	tmp = tmp ^ rotater64(tmp, 1)
	return 0xfffffffffffffffc ^ i ^ uint64(z) ^ tmp
}

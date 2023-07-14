// asmcheck

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Make sure this code doesn't generate spill/restore.

package codegen

func pack20(in *[20]uint64) uint64 {
	var out uint64
	out |= 4
	// amd64:-`.*SP.*`
	out |= in[0] << 4
	// amd64:-`.*SP.*`
	out |= in[1] << 7
	// amd64:-`.*SP.*`
	out |= in[2] << 10
	// amd64:-`.*SP.*`
	out |= in[3] << 13
	// amd64:-`.*SP.*`
	out |= in[4] << 16
	// amd64:-`.*SP.*`
	out |= in[5] << 19
	// amd64:-`.*SP.*`
	out |= in[6] << 22
	// amd64:-`.*SP.*`
	out |= in[7] << 25
	// amd64:-`.*SP.*`
	out |= in[8] << 28
	// amd64:-`.*SP.*`
	out |= in[9] << 31
	// amd64:-`.*SP.*`
	out |= in[10] << 34
	// amd64:-`.*SP.*`
	out |= in[11] << 37
	// amd64:-`.*SP.*`
	out |= in[12] << 40
	// amd64:-`.*SP.*`
	out |= in[13] << 43
	// amd64:-`.*SP.*`
	out |= in[14] << 46
	// amd64:-`.*SP.*`
	out |= in[15] << 49
	// amd64:-`.*SP.*`
	out |= in[16] << 52
	// amd64:-`.*SP.*`
	out |= in[17] << 55
	// amd64:-`.*SP.*`
	out |= in[18] << 58
	// amd64:-`.*SP.*`
	out |= in[19] << 61
	return out
}

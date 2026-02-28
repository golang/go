// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/cpu"
)

var memmoveBits uint8

const (
	// avxSupported indicates that the CPU supports AVX instructions.
	avxSupported = 1 << 0

	// repmovsPreferred indicates that REP MOVSx instruction is more
	// efficient on the CPU.
	repmovsPreferred = 1 << 1
)

func init() {
	// Here we assume that on modern CPUs with both FSRM and ERMS features,
	// copying data blocks of 2KB or larger using the REP MOVSB instruction
	// will be more efficient to avoid having to keep up with CPU generations.
	// Therefore, we may retain a BlockList mechanism to ensure that microarchitectures
	// that do not fit this case may appear in the future.
	// We enable it on Intel CPUs first, and we may support more platforms
	// in the future.
	isERMSNiceCPU := isIntel
	useREPMOV := isERMSNiceCPU && cpu.X86.HasERMS && cpu.X86.HasFSRM
	if cpu.X86.HasAVX {
		memmoveBits |= avxSupported
	}
	if useREPMOV {
		memmoveBits |= repmovsPreferred
	}
}

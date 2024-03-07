// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Create a large frame to force stack growth. See #62326.
TEXT ·testSPWrite(SB),0,$16384-0
	// Write to SP
	MOVQ SP, AX
	ANDQ $~0xf, SP
	MOVQ AX, SP
	RET

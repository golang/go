// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !amd64 && !arm64 && !loong64

#include "textflag.h"

// func block(counter uint64, seed *[8]uint32, blocks *[16][4]uint32)
TEXT ·block(SB), NOSPLIT, $0
	JMP ·block_generic(SB)


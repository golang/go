// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

TEXT ·sigill(SB),0,$0-0
	UD2	// generates a SIGILL
	RET

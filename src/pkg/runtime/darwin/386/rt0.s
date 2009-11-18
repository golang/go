// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Darwin and Linux use the same linkage to main

TEXT	_rt0_386_darwin(SB),7,$0
	JMP	_rt0_386(SB)

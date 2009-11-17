// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Darwin and Linux use the same linkage to main

TEXT	_rt0_amd64_freebsd(SB),7,$-8
	MOVQ	$_rt0_amd64(SB), DX
	JMP	DX

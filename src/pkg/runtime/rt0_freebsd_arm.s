// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// FreeBSD and Linux use the same linkage to main

TEXT _rt0_arm_freebsd(SB),7,$-4
	B	_rt0_arm(SB)

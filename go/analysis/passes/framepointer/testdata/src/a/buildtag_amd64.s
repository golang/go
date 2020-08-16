// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build nope

TEXT ·bt1(SB), 0, $0
	MOVQ	$0, BP // ok because of build tag
	RET

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

TEXT	·newcoro(SB),0,$0-0
	CALL	runtime·newcoro(SB)
	RET

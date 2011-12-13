// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

TEXT ·Floor(SB),7,$0
	B	·floor(SB)

TEXT ·Ceil(SB),7,$0
	B	·ceil(SB)

TEXT ·Trunc(SB),7,$0
	B	·trunc(SB)

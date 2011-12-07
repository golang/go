// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// no memchr implementation on arm yet
TEXT ·IndexByte(SB),7,$0
	B	·indexBytePortable(SB)

// no memcmp implementation on arm yet
TEXT ·Equal(SB),7,$0
	B	·equalPortable(SB)

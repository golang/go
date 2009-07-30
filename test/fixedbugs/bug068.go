// errchk $G $D/$F.go

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// RESOLUTION: This program is illegal.  We should reject all unnecessary backslashes.

package main

const c = '\'';  // this works
const s = "\'";  // ERROR "invalid|escape"

/*
There is no reason why the escapes need to be different inside strings and chars.

uetli:~/go/test/bugs gri$ 6g bug068.go
bug068.go:6: unknown escape sequence: '
*/

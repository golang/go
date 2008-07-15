// errchk $G $D/$F.go

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

const c = '\'';  // this works
const s = "\'";  // this doesn't

/*
There is no reason why the escapes need to be different inside strings and chars.

uetli:~/go/test/bugs gri$ 6g bug065.go
bug065.go:6: unknown escape sequence: '
*/

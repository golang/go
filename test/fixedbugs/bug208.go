// errorcheck

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type	T struct
{
	f int;
}

// 6g used to get confused by the f:1 above
// and allow uses of f that would be silently
// dropped during the compilation.
var _ = f;	// ERROR "undefined"

var _ = T{f: 1}


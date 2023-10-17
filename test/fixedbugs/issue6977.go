// errorcheck

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

import "io"

// Alan's initial report.

type I interface { f(); String() string }
type J interface { g(); String() string }

type IJ1 = interface { I; J }
type IJ2 = interface { f(); g(); String() string }

var _ = (*IJ1)(nil) == (*IJ2)(nil) // static assert that IJ1 and IJ2 are identical types

// The canonical example.

type ReadWriteCloser interface { io.ReadCloser; io.WriteCloser }

// Some more cases.

type M interface { m() }
type M32 interface { m() int32 }
type M64 interface { m() int64 }

type U1 interface { m() }
type U2 interface { m(); M }
type U3 interface { M; m() }
type U4 interface { M; M; M }
type U5 interface { U1; U2; U3; U4 }

type U6 interface { m(); m() } // ERROR "duplicate method .*m"
type U7 interface { M32; m() } // ERROR "duplicate method .*m"
type U8 interface { m(); M32 } // ERROR "duplicate method .*m"
type U9 interface { M32; M64 } // ERROR "duplicate method .*m"

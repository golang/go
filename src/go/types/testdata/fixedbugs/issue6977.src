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

type U6 interface { m(); m /* ERROR duplicate method */ () }
type U7 interface { M32 /* ERROR duplicate method */ ; m() }
type U8 interface { m(); M32 /* ERROR duplicate method */ }
type U9 interface { M32; M64 /* ERROR duplicate method */ }

// Verify that repeated embedding of the same interface(s)
// eliminates duplicate methods early (rather than at the
// end) to prevent exponential memory and time use.
// Without early elimination, computing T29 may take dozens
// of minutes.
type (
        T0 interface { m() }
        T1 interface { T0; T0 }
        T2 interface { T1; T1 }
        T3 interface { T2; T2 }
        T4 interface { T3; T3 }
        T5 interface { T4; T4 }
        T6 interface { T5; T5 }
        T7 interface { T6; T6 }
        T8 interface { T7; T7 }
        T9 interface { T8; T8 }

        T10 interface { T9; T9 }
        T11 interface { T10; T10 }
        T12 interface { T11; T11 }
        T13 interface { T12; T12 }
        T14 interface { T13; T13 }
        T15 interface { T14; T14 }
        T16 interface { T15; T15 }
        T17 interface { T16; T16 }
        T18 interface { T17; T17 }
        T19 interface { T18; T18 }

        T20 interface { T19; T19 }
        T21 interface { T20; T20 }
        T22 interface { T21; T21 }
        T23 interface { T22; T22 }
        T24 interface { T23; T23 }
        T25 interface { T24; T24 }
        T26 interface { T25; T25 }
        T27 interface { T26; T26 }
        T28 interface { T27; T27 }
        T29 interface { T28; T28 }
)

// Verify that m is present.
var x T29
var _ = x.m

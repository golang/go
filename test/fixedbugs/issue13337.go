// compile

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 13337: The Go compiler limited how deeply embedded types
// were searched for promoted fields and methods.

package s

type S0 struct{ f int }
func (S0) m() {}

type S1 struct{ S0 }
type S2 struct{ S1 }
type S3 struct{ S2 }
type S4 struct{ S3 }
type S5 struct{ S4 }
type S6 struct{ S5 }
type S7 struct{ S6 }
type S8 struct{ S7 }
type S9 struct{ S8 }
type S10 struct{ S9 }
type S11 struct{ S10 }
type S12 struct{ S11 }
type S13 struct{ S12 }

var _ = S13{}.f
var _ = S13.m

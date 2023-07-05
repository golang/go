// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type T0[P any, P1 T0[P,P1]] any
type T1[P any, P1 *T1[P,P1]] any


//TODO: current implementation doesn't allow this due to lack of applications for this syntax.
// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type T0[P0 T2, P1 T1[int]] interface{}
type T1 /* ERROR "invalid recursive type" */ [P0 T2] T1[int]
type T2 T0[T2, int]

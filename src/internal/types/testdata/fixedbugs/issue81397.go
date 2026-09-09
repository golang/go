// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func f[P, Q any]() {}

// Don't panic on the next line, produce a proper error instead.
var _ any = f /* ERROR "cannot infer Q" */ [int]

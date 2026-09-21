// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type List[P /* ERROR "instantiation cycle" */ any] struct{}

func (_ List[P]) m() (_ List[List[P]]) { return }

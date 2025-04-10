// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !math_big_pure_go

package big

import "internal/cpu"

var hasVX = cpu.S390X.HasVX

func addVVvec(z, x, y []Word) (c Word)
func subVVvec(z, x, y []Word) (c Word)

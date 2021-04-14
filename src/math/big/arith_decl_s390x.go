// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !math_big_pure_go

package big

import "internal/cpu"

func addVV_check(z, x, y []Word) (c Word)
func addVV_vec(z, x, y []Word) (c Word)
func addVV_novec(z, x, y []Word) (c Word)
func subVV_check(z, x, y []Word) (c Word)
func subVV_vec(z, x, y []Word) (c Word)
func subVV_novec(z, x, y []Word) (c Word)

var hasVX = cpu.S390X.HasVX

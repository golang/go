// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func _(a int, b /* ERROR missing parameter type */ )
func _(a int, /* ERROR missing parameter name */ []int)
func _(a int, /* ERROR missing parameter name */ []int, c int)

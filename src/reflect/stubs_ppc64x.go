// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ppc64le || ppc64

package reflect

func archFloat32FromReg(reg uint64) float32
func archFloat32ToReg(val float32) uint64

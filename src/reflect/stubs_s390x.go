// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build s390x

package reflect

func archFloat32FromReg(reg uint64) float32
func archFloat32ToReg(val float32) uint64

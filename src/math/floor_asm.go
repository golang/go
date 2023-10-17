// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build 386 || amd64 || arm64 || ppc64 || ppc64le || s390x || wasm

package math

const haveArchFloor = true

func archFloor(x float64) float64

const haveArchCeil = true

func archCeil(x float64) float64

const haveArchTrunc = true

func archTrunc(x float64) float64

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build gc

package cpu

func getisar0() uint64
func getisar1() uint64
func getmmfr1() uint64
func getpfr0() uint64
func getzfr0() uint64

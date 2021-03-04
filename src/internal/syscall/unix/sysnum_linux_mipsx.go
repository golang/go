// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build mips || mipsle
// +build mips mipsle

package unix

const (
	getrandomTrap     uintptr = 4353
	copyFileRangeTrap uintptr = 4360
)

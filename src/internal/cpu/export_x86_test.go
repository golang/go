// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build 386 || amd64

package cpu

var (
	GetGOAMD64level = getGOAMD64level
)

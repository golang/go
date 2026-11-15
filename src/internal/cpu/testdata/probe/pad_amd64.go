// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64

package main

import "internal/cpu"

var _ [64 - cpu.CacheLinePadSize]byte
var _ [cpu.CacheLinePadSize - 64]byte

func main() {}

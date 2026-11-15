// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build arm64

package main

import "internal/cpu"

var _ [128 - cpu.CacheLinePadSize]byte
var _ [cpu.CacheLinePadSize - 128]byte

func main() {}

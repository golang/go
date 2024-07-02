// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build windows

package getrand

import (
	"internal/syscall/windows"
	"math"
)

const maxGetRandomRead = math.MaxInt

func getRandom(b []byte) error {
	return windows.ProcessPrng(b)
}

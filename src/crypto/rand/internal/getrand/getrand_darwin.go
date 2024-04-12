// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin

package getrand

import (
	"internal/syscall/unix"
	"math"
)

const maxGetRandomRead = math.MaxInt

func getRandom(out []byte) error {
	unix.ARC4Random(out)
	return nil
}

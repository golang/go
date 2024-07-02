// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux || dragonfly || freebsd || illumos || solaris || darwin || openbsd || netbsd || (js && wasm) || wasip1 || windows

package getrand

import "math"

// GetRandom populates out with cryptographically secure random data.
func GetRandom(out []byte) error {
	if maxGetRandomRead == math.MaxInt {
		return getRandom(out)
	}

	// Batch random read operations up to maxGetRandomRead.
	for len(out) > 0 {
		readBytes := min(len(out), maxGetRandomRead)
		if err := getRandom(out[:readBytes]); err != nil {
			return err
		}
		out = out[readBytes:]
	}
	return nil
}

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux || dragonfly || freebsd || illumos || solaris || darwin || openbsd || netbsd || (js && wasm) || wasip1 || windows

package getrand

import "math"

func GetRandom(out []byte) error {
	if maxGetRandomRead == math.MaxInt {
		return getRandom(out)
	}

	for len(out) > 0 {
		read := len(out)
		if read > maxGetRandomRead {
			read = maxGetRandomRead
		}
		if err := getRandom(out[:read]); err != nil {
			return err
		}
		out = out[read:]
	}
	return nil
}

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

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

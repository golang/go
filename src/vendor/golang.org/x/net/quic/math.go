// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package quic

func abs[T ~int | ~int64](a T) T {
	if a < 0 {
		return -a
	}
	return a
}

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

func D(_ string, _ int) (uint64, string) {
	return 101, "bad"
}

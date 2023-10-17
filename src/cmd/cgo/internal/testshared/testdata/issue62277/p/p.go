// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

var S = func() []string {
	return []string{"LD_LIBRARY_PATH"}
}()

var T []string

func init() {
	T = func() []string {
		return []string{"LD_LIBRARY_PATH"}
	}()
}

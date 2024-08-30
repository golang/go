// compile

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func f() {
	for b := "" < join([]string{}, "") && true; ; {
		_ = b
	}
}

//go:noinline
func join(elems []string, sep string) string {
	return ""
}

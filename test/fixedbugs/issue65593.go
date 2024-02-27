// compile

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

const run = false

func f() {
	if !run {
		return
	}

	messages := make(chan struct{}, 1)
main:
	for range messages {
		break main
	}
}

// compile

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 30430: isGoConst returned true for non-const variables,
// resulting in ICE.

package p

func f() {
	var s string
	_ = map[string]string{s: ""}
}

const s = ""

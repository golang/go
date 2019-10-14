// compile

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Triggers a double walk of the (inlined) switch in il

package p

func il(s string) string {
	switch len(s) {
	case 0:
		return "zero"
	case 1:
		return "one"
	}
	return s
}

func f() {
	var s string
	var as []string
	switch false && (s+"a"+as[0]+il(s)+as[0]+s == "") {
	}
}

// compile

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// gofrontend incorrectly gave an error for this code.

package p

type B bool

func main() {
	var v B = false
	if (true && true) && v {
	}
}

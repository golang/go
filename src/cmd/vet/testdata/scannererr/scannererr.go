// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package scannererr

import (
	"bufio"
	"io"
)

func _(r io.Reader) {
	s := bufio.NewScanner(r) // ERROR `bufio.Scanner .* is used in Scan loop at line 14 without final check of s.Err\(\)`
	for s.Scan() {
	}
}

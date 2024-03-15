// -lang=go1.16

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.21

package main

import "slices"

func main() {
	_ = slices.Clone([]string{}) // no error should be reported here
}

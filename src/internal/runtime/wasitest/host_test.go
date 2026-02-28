// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package wasi_test

import "flag"

var target string

func init() {
	// The dist test runner passes -target when running this as a host test.
	flag.StringVar(&target, "target", "", "")
}

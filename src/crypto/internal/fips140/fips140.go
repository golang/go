// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fips140

import "crypto/internal/fips140deps/godebug"

var Enabled bool

var debug bool

func init() {
	switch godebug.Value("#fips140") {
	case "on", "only":
		Enabled = true
	case "debug":
		Enabled = true
		debug = true
	}
}

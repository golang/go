// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fips

import "internal/godebug"

var Enabled bool

func init() {
	switch godebug.New("#fips140").Value() {
	case "on", "debug", "only":
		Enabled = true
	}
}

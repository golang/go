// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !logosejected

package rt

import _ "unsafe"

//go:linkname logosRegionBytes runtime.logosRegionBytes
func logosRegionBytes() int

func RegionBytes() int {
	return logosRegionBytes()
}

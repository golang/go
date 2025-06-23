// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/cpu"
)

var arm64UseAlignedLoads bool

func init() {
	if cpu.ARM64.IsNeoverse {
		arm64UseAlignedLoads = true
	}
}

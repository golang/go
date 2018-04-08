// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"internal/cpu"
)

var hasGCMAsm = cpu.ARM64.HasAES && cpu.ARM64.HasPMULL


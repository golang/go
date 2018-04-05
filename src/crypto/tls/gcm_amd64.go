// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"internal/cpu"
)

func hasGCMAsm() bool {
	return cpu.X86.HasAES && cpu.X86.HasPCLMULQDQ 
}

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package sysinfo implements high level hardware information gathering
// that can be used for debugging or information purposes.
package sysinfo

import (
	"internal/cpu"
	"sync"
)

var CPUName = sync.OnceValue(func {
	if name := cpu.Name(); name != "" {
		return name
	}

	if name := osCPUInfoName(); name != "" {
		return name
	}

	return ""
})

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

var cpuInfo struct {
	once sync.Once
	name string
}

func CPUName() string {
	cpuInfo.once.Do(func() {
		// Try to get the information from internal/cpu.
		if name := cpu.Name(); name != "" {
			cpuInfo.name = name
			return
		}

		// TODO(martisch): use /proc/cpuinfo and /sys/devices/system/cpu/ on Linux as fallback.
		if name := osCpuInfoName(); name != "" {
			cpuInfo.name = name
			return
		}
	})

	return cpuInfo.name
}

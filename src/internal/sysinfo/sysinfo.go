// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package sysinfo implements high level hardware information gathering
// that can be used for debugging or information purposes.
package sysinfo

import (
	internalcpu "internal/cpu"
	"sync"
)

type cpuInfo struct {
	once sync.Once
	name string
}

var CPU cpuInfo

func (cpu *cpuInfo) Name() string {
	cpu.once.Do(func() {
		// Try to get the information from internal/cpu.
		if name := internalcpu.Name(); name != "" {
			cpu.name = name
			return
		}
		// TODO(martisch): use /proc/cpuinfo and /sys/devices/system/cpu/ on Linux as fallback.
	})
	return cpu.name
}

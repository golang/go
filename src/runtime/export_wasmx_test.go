// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build wasm || wasm32

package runtime

var SysReserveAlignedSbrk = sysReserveAlignedSbrk

// GCAssistUpdatesLimiter must be called with GC disabled. On WebAssembly
// there is no sysmon to update the limiter between setup and the assist.
func GCAssistUpdatesLimiter() bool {
	gcCPULimiter.update(nanotime())
	fill := gcCPULimiter.bucket.fill
	enabled := gcCPULimiter.enabled.Load()
	lastUpdate := gcCPULimiter.lastUpdate.Load()
	defer func() {
		gcCPULimiter.bucket.fill = fill
		gcCPULimiter.enabled.Store(enabled)
		gcCPULimiter.lastUpdate.Store(lastUpdate)
	}()
	before := nanotime() - gcCPULimiterUpdatePeriod - 1
	gcCPULimiter.bucket.fill = gcCPULimiter.bucket.capacity
	gcCPULimiter.lastUpdate.Store(before)
	gcCPULimiter.enabled.Store(true)
	gp := getg()
	debt := gp.gcAssistBytes
	gp.gcAssistBytes = 0
	gcAssistAlloc(gp)
	gp.gcAssistBytes = debt
	return gcCPULimiter.lastUpdate.Load() != before && !gcCPULimiter.limiting()
}

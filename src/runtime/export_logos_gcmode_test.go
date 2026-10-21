// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

// LogosDefaultHeapMinimum is the heap minimum the batch trigger floor uses.
const LogosDefaultHeapMinimum = defaultHeapMinimum

// LogosBatchLimit exposes logosBatchLimit.
func LogosBatchLimit(memoryLimit int64, phys uint64) uint64 {
	return logosBatchLimit(memoryLimit, phys)
}

// LogosBatchTrigger exposes logosBatchTriggerAt.
func LogosBatchTrigger(limit, heapMarked uint64) uint64 {
	return logosBatchTriggerAt(limit, heapMarked)
}

// LogosModeRefuses reports which triggers mode refuses: the heap trigger, the
// periodic trigger, and the explicit runtime.GC trigger.
func LogosModeRefuses(mode string) (heap, periodic, explicit bool) {
	return logosModeRefuses(mode, gcTrigger{kind: gcTriggerHeap}),
		logosModeRefuses(mode, gcTrigger{kind: gcTriggerTime}),
		logosModeRefuses(mode, gcTrigger{kind: gcTriggerCycle})
}

// LogosSanePhysMem exposes logosSanePhysMem.
func LogosSanePhysMem(phys uint64) uint64 {
	return logosSanePhysMem(phys)
}

// LogosParseUint exposes logosParseUint.
func LogosParseUint(b []byte) (uint64, int) {
	return logosParseUint(b)
}

// LogosPhysicalMemory exposes logosPhysicalMemory.
func LogosPhysicalMemory() uint64 {
	return logosPhysicalMemory()
}

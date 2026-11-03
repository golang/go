// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jit

package jit

import (
	"fmt"
	"io"
	"unsafe"
)

// MetadataError identifies a structurally invalid region or safepoint. Validation
// cannot prove that machine code matches its maps, detect omitted live pointers,
// or prove that a claimed pointer word was initialized. Those remain JIT duties.
type MetadataError struct {
	Start, End, PCOffset uintptr
	Field, Reason        string
}

func (e *MetadataError) Error() string {
	return fmt.Sprintf("runtime/jit: region [%#x,%#x) PC+%#x: %s: %s", e.Start, e.End, e.PCOffset, e.Field, e.Reason)
}

// ValidateRegion checks metadata without publishing it or calling Describe.
// Register and AddStackMaps perform the same checks and panic with MetadataError
// before invoking runtime registration. Overlap and handle lifetime are checked
// separately under the publication lock.
func ValidateRegion(r Region) error { return validateMaps(r, r.StackMaps, 0, false) }

func validateMaps(r Region, maps []StackMap, previous uintptr, hasPrevious bool) error {
	bad := func(pc uintptr, field, why string) error { return &MetadataError{r.Start, r.End, pc, field, why} }
	if r.Start == 0 || r.End <= r.Start {
		return bad(0, "range", "expected nonempty executable range")
	}
	if r.Unwind > UnwindDeclare {
		return bad(0, "Unwind", "unknown mode")
	}
	ptr := unsafe.Sizeof(uintptr(0))
	for _, m := range maps {
		if m.PCOffset >= r.End-r.Start || hasPrevious && m.PCOffset <= previous {
			return bad(m.PCOffset, "PCOffset", "outside region or not strictly increasing")
		}
		if m.FrameOffset%ptr != 0 {
			return bad(m.PCOffset, "FrameOffset", "not pointer aligned")
		}
		if m.FrameWords > uintptr(^uint32(0)>>1) || m.FrameWords > (^uintptr(0)-m.FrameOffset)/ptr {
			return bad(m.PCOffset, "FrameWords", "range overflow")
		}
		bytes := (m.FrameWords + 7) / 8
		if uintptr(len(m.PointerMask)) < bytes {
			return bad(m.PCOffset, "PointerMask", "short bitmap")
		}
		if m.FrameWords%8 != 0 && bytes > 0 && m.PointerMask[bytes-1]>>uint(m.FrameWords%8) != 0 {
			return bad(m.PCOffset, "PointerMask", "bits outside frame")
		}
		if r.Unwind != UnwindStop && !m.HasUnwind {
			return bad(m.PCOffset, "HasUnwind", "traversable frames require an unwind recipe")
		}
		if m.HasUnwind {
			if m.UnwindBaseOffset > ^uintptr(0)-ptr {
				return bad(m.PCOffset, "UnwindBaseOffset", "overflowing unwind base")
			}

			for _, f := range []struct {
				name string
				v    uintptr
			}{{"UnwindBaseOffset", m.UnwindBaseOffset}, {"CallerPCOffset", m.CallerPCOffset}, {"CallerSPOffset", m.CallerSPOffset}, {"CallerBPOffset", m.CallerBPOffset}} {
				if f.v%ptr != 0 || f.v > ^uintptr(0)-m.UnwindBaseOffset-ptr {
					return bad(m.PCOffset, f.name, "unaligned or overflowing unwind address")
				}
			}
			if m.CallerSPOffset == 0 && m.UnwindBaseOffset == 0 && !m.UnwindBaseUsesDelta {
				return bad(m.PCOffset, "CallerSPOffset", "unwind must advance the stack")
			}
			if m.UnwindBaseUsesDelta && (m.UnwindBaseDeltaOffset%ptr != 0 || m.UnwindBaseDeltaOffset > ^uintptr(0)-ptr) {
				return bad(m.PCOffset, "UnwindBaseDeltaOffset", "unaligned or overflowing delta address")
			}
		}
		previous, hasPrevious = m.PCOffset, true
	}
	return nil
}

func copyMaps(maps []StackMap) []StackMap {
	out := append([]StackMap(nil), maps...)
	for i := range out {
		out[i].PointerMask = append([]byte(nil), out[i].PointerMask...)
	}
	return out
}

// Snapshot returns an independent diagnostic copy of the published region's
// metadata. It does not execute Describe. Unregistered handles panic.
func (h Handle) Snapshot() Region {
	liveRegionsMu.Lock()
	defer liveRegionsMu.Unlock()
	r, ok := liveRegions[h.handle]
	if !ok {
		panic("runtime/jit: unregistered handle")
	}
	r.StackMaps = copyMaps(r.StackMaps)
	return r
}

// WriteTo writes registered PCs, root slots, and unwind recipes for diagnosis.
// It runs in ordinary Go context, may allocate, and must not be called from
// Describe or GC callbacks. The returned count follows io.WriterTo.
func (h Handle) WriteTo(w io.Writer) (n int64, err error) {
	r := h.Snapshot()
	write := func(format string, args ...any) {
		if err != nil {
			return
		}
		var count int
		count, err = fmt.Fprintf(w, format, args...)
		n += int64(count)
	}
	write("region [%#x,%#x) unwind=%d\n", r.Start, r.End, r.Unwind)
	for _, m := range r.StackMaps {
		write("pc=%#x (+%#x) frame=SP+%d words=%d roots=%x unwind=%t base=SP+%d delta=%t@%d callerPC=base+%d callerSP=base+%d callerBP=base+%d\n", r.Start+m.PCOffset, m.PCOffset, m.FrameOffset, m.FrameWords, m.PointerMask, m.HasUnwind, m.UnwindBaseOffset, m.UnwindBaseUsesDelta, m.UnwindBaseDeltaOffset, m.CallerPCOffset, m.CallerSPOffset, m.CallerBPOffset)
	}
	return
}

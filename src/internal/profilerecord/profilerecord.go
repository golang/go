// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package profilerecord holds internal types used to represent profiling
// records with deep stack traces.
//
// TODO: Consider moving this to internal/runtime, see golang.org/issue/65355.
package profilerecord

type StackRecord struct {
	Stack []uintptr
}

type MemProfileRecord struct {
	ObjectSize                int64
	AllocObjects, FreeObjects int64
	Stack                     []uintptr
}

func (r *MemProfileRecord) InUseBytes() int64   { return r.InUseObjects() * r.ObjectSize }
func (r *MemProfileRecord) InUseObjects() int64 { return r.AllocObjects - r.FreeObjects }

type BlockProfileRecord struct {
	Count  int64
	Cycles int64
	Stack  []uintptr
}

// MemProfileMetrics holds runtime metrics associated with the most
// recently captured inuse heap profile.
//
// The fields whose names mirror runtime/metrics paths are populated from
// the same underlying sources as the corresponding metrics in the
// runtime/metrics package; see that package for documentation of their
// meaning. Heap-derived values are captured from the heap statistics
// snapshot taken at mark termination for the profile cycle. Other values
// are captured when the active heap profile is published.
type MemProfileMetrics struct {
	// RSS is the process resident set size in bytes at the moment the
	// active heap profile was published, including pages shared with
	// other processes. Zero means RSS was unavailable.
	RSS uint64

	// HeapLive corresponds to /gc/heap/live:bytes: the bytes marked
	// live by the most recent GC.
	HeapLive uint64

	// MemoryClassesHeapFree corresponds to /memory/classes/heap/free:bytes.
	MemoryClassesHeapFree uint64
	// MemoryClassesHeapObjects corresponds to /memory/classes/heap/objects:bytes.
	MemoryClassesHeapObjects uint64
	// MemoryClassesHeapReleased corresponds to /memory/classes/heap/released:bytes.
	MemoryClassesHeapReleased uint64
	// MemoryClassesHeapStacks corresponds to /memory/classes/heap/stacks:bytes.
	MemoryClassesHeapStacks uint64
	// MemoryClassesHeapUnused corresponds to /memory/classes/heap/unused:bytes.
	MemoryClassesHeapUnused uint64

	// MemoryClassesMetadataMCacheFree corresponds to
	// /memory/classes/metadata/mcache/free:bytes.
	MemoryClassesMetadataMCacheFree uint64
	// MemoryClassesMetadataMCacheInuse corresponds to
	// /memory/classes/metadata/mcache/inuse:bytes.
	MemoryClassesMetadataMCacheInuse uint64
	// MemoryClassesMetadataMSpanFree corresponds to
	// /memory/classes/metadata/mspan/free:bytes.
	MemoryClassesMetadataMSpanFree uint64
	// MemoryClassesMetadataMSpanInuse corresponds to
	// /memory/classes/metadata/mspan/inuse:bytes.
	MemoryClassesMetadataMSpanInuse uint64
	// MemoryClassesMetadataOther corresponds to
	// /memory/classes/metadata/other:bytes.
	MemoryClassesMetadataOther uint64

	// MemoryClassesOSStacks corresponds to /memory/classes/os-stacks:bytes.
	MemoryClassesOSStacks uint64
	// MemoryClassesOther corresponds to /memory/classes/other:bytes.
	MemoryClassesOther uint64
	// MemoryClassesProfilingBuckets corresponds to
	// /memory/classes/profiling/buckets:bytes.
	MemoryClassesProfilingBuckets uint64
	// MemoryClassesTotal corresponds to /memory/classes/total:bytes.
	MemoryClassesTotal uint64
}

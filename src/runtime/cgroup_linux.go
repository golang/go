// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/runtime/cgroup"
)

// cgroup-aware GOMAXPROCS default
//
// At startup (defaultGOMAXPROCSInit), we read /proc/self/cgroup and /proc/self/mountinfo
// to find our current CPU cgroup and open its limit file(s), which remain open
// for the entire process lifetime. We periodically read the current limit by
// rereading the limit file(s) from the beginning.
//
// This makes reading updated limits simple, but has a few downsides:
//
// 1. We only read the limit from the leaf cgroup that actually contains this
// process. But a parent cgroup may have a tighter limit. That tighter limit
// would be our effective limit. That said, container runtimes tend to hide
// parent cgroups from the container anyway.
//
// 2. If the process is migrated to another cgroup while it is running it will
// not notice, as we only check which cgroup we are in once at startup.
var (
	// We can't allocate during early initialization when we need to find
	// the cgroup. Simply use a fixed global as a scratch parsing buffer.
	cgroupScratch [cgroup.ScratchSize]byte

	cgroupOK  bool
	cgroupCPU cgroup.CPU

	// defaultGOMAXPROCSInit runs before internal/godebug init, so we can't
	// directly update the GODEBUG counter. Store the result until after
	// init runs.
	containermaxprocsNonDefault bool
	containermaxprocs           = &godebugInc{name: "containermaxprocs"}
)

// Prepare for defaultGOMAXPROCS.
//
// Must run after parsedebugvars.
func defaultGOMAXPROCSInit() {
	c, err := cgroup.OpenCPU(cgroupScratch[:])
	if err != nil {
		// Likely cgroup.ErrNoCgroup.
		return
	}

	if debug.containermaxprocs > 0 {
		// Normal operation.
		cgroupCPU = c
		cgroupOK = true
		return
	}

	// cgroup-aware GOMAXPROCS is disabled. We still check the cgroup once
	// at startup to see if enabling the GODEBUG would result in a
	// different default GOMAXPROCS. If so, we increment runtime/metrics
	// /godebug/non-default-behavior/cgroupgomaxprocs:events.
	procs := getCPUCount()
	cgroupProcs := adjustCgroupGOMAXPROCS(procs, c)
	if procs != cgroupProcs {
		containermaxprocsNonDefault = true
	}

	// Don't need the cgroup for remaining execution.
	c.Close()
}

// defaultGOMAXPROCSUpdateGODEBUG updates the internal/godebug counter for
// container GOMAXPROCS, once internal/godebug is initialized.
func defaultGOMAXPROCSUpdateGODEBUG() {
	if containermaxprocsNonDefault {
		containermaxprocs.IncNonDefault()
	}
}

// Return the default value for GOMAXPROCS when it has not been set explicitly.
//
// ncpu is the optional precomputed value of getCPUCount. If passed as 0,
// defaultGOMAXPROCS will call getCPUCount.
func defaultGOMAXPROCS(ncpu int32) int32 {
	// GOMAXPROCS is the minimum of:
	//
	// 1. Total number of logical CPUs available from sched_getaffinity.
	//
	// 2. The average CPU cgroup throughput limit (average throughput =
	// quota/period). A limit less than 2 is rounded up to 2, and any
	// fractional component is rounded up.
	//
	// TODO: add rationale.

	procs := ncpu
	if procs <= 0 {
		procs = getCPUCount()
	}
	if !cgroupOK {
		// No cgroup, or disabled by debug.containermaxprocs.
		return procs
	}

	return adjustCgroupGOMAXPROCS(procs, cgroupCPU)
}

// Lower procs as necessary for the current cgroup CPU limit.
func adjustCgroupGOMAXPROCS(procs int32, cpu cgroup.CPU) int32 {
	limit, ok, err := cgroup.ReadCPULimit(cpu)
	if err == nil && ok {
		limit = ceil(limit)
		limit = max(limit, 2)
		if int32(limit) < procs {
			procs = int32(limit)
		}
	}
	return procs
}

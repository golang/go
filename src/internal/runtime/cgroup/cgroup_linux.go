// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgroup

import (
	"internal/runtime/syscall/linux"
)

// Include explicit NUL to be sure we include it in the slice.
const (
	v2MaxFile    = "/cpu.max\x00"
	v1QuotaFile  = "/cpu.cfs_quota_us\x00"
	v1PeriodFile = "/cpu.cfs_period_us\x00"
)

// CPU owns the FDs required to read the CPU limit from a cgroup.
type CPU struct {
	version Version

	// For cgroup v1, this is cpu.cfs_quota_us.
	// For cgroup v2, this is cpu.max.
	quotaFD int

	// For cgroup v1, this is cpu.cfs_period_us.
	// For cgroup v2, this is unused.
	periodFD int
}

func (c CPU) Close() {
	switch c.version {
	case V1:
		linux.Close(c.quotaFD)
		linux.Close(c.periodFD)
	case V2:
		linux.Close(c.quotaFD)
	default:
		throw("impossible cgroup version")
	}
}

func checkBufferSize(s []byte, size int) {
	if len(s) != size {
		println("runtime: cgroup buffer length", len(s), "want", size)
		throw("runtime: cgroup invalid buffer length")
	}
}

// OpenCPU returns a CPU for the CPU cgroup containing the current process, or
// ErrNoCgroup if the process is not in a CPU cgroup.
//
// scratch must have length ScratchSize.
func OpenCPU(scratch []byte) (CPU, error) {
	checkBufferSize(scratch, ScratchSize)

	base := scratch[:PathSize]
	scratch2 := scratch[PathSize:]

	n, version, err := FindCPU(base, scratch2)
	if err != nil {
		return CPU{}, err
	}

	switch version {
	case 1:
		n2 := copy(base[n:], v1QuotaFile)
		path := base[:n+n2]
		quotaFD, errno := linux.Open(&path[0], linux.O_RDONLY|linux.O_CLOEXEC, 0)
		if errno != 0 {
			// This may fail if this process was migrated out of
			// the cgroup found by FindCPU and that cgroup has been
			// deleted.
			return CPU{}, errSyscallFailed
		}

		n2 = copy(base[n:], v1PeriodFile)
		path = base[:n+n2]
		periodFD, errno := linux.Open(&path[0], linux.O_RDONLY|linux.O_CLOEXEC, 0)
		if errno != 0 {
			// This may fail if this process was migrated out of
			// the cgroup found by FindCPU and that cgroup has been
			// deleted.
			return CPU{}, errSyscallFailed
		}

		c := CPU{
			version:  1,
			quotaFD:  quotaFD,
			periodFD: periodFD,
		}
		return c, nil
	case 2:
		n2 := copy(base[n:], v2MaxFile)
		path := base[:n+n2]
		maxFD, errno := linux.Open(&path[0], linux.O_RDONLY|linux.O_CLOEXEC, 0)
		if errno != 0 {
			// This may fail if this process was migrated out of
			// the cgroup found by FindCPU and that cgroup has been
			// deleted.
			return CPU{}, errSyscallFailed
		}

		c := CPU{
			version:  2,
			quotaFD:  maxFD,
			periodFD: -1,
		}
		return c, nil
	default:
		throw("impossible cgroup version")
		panic("unreachable")
	}
}

// Returns average CPU throughput limit from the cgroup, or ok false if there
// is no limit.
func ReadCPULimit(c CPU) (float64, bool, error) {
	switch c.version {
	case 1:
		quota, err := readV1Number(c.quotaFD)
		if err != nil {
			return 0, false, errMalformedFile
		}

		if quota < 0 {
			// No limit.
			return 0, false, nil
		}

		period, err := readV1Number(c.periodFD)
		if err != nil {
			return 0, false, errMalformedFile
		}

		return float64(quota) / float64(period), true, nil
	case 2:
		// quotaFD is the cpu.max FD.
		return readV2Limit(c.quotaFD)
	default:
		throw("impossible cgroup version")
		panic("unreachable")
	}
}

// Returns the value from the quota/period file.
func readV1Number(fd int) (int64, error) {
	// The format of the file is "<value>\n" where the value is in
	// int64 microseconds and, if quota, may be -1 to indicate no limit.
	//
	// MaxInt64 requires 19 bytes to display in base 10, thus the
	// conservative max size of this file is 19 + 1 (newline) = 20 bytes.
	// We'll provide a bit more for good measure.
	//
	// Always read from the beginning of the file to get a fresh value.
	var b [64]byte
	n, errno := linux.Pread(fd, b[:], 0)
	if errno != 0 {
		return 0, errSyscallFailed
	}
	if n == len(b) {
		return 0, errMalformedFile
	}

	buf := b[:n]
	return parseV1Number(buf)
}

// Returns CPU throughput limit, or ok false if there is no limit.
func readV2Limit(fd int) (float64, bool, error) {
	// The format of the file is "<quota> <period>\n" where quota and
	// period are microseconds and quota may be "max" to indicate no limit.
	//
	// Note that the kernel is inconsistent about whether the values are
	// uint64 or int64: values are parsed as uint64 but printed as int64.
	// See kernel/sched/core.c:cpu_max_{show,write}.
	//
	// In practice, the kernel limits the period to 1s (1000000us) (see
	// max_cfs_quota_period), and the quota to (1<<44)us (see
	// max_cfs_runtime), so these values can't get large enough for the
	// distinction to matter.
	//
	// MaxInt64 requires 19 bytes to display in base 10, thus the
	// conservative max size of this file is 19 + 19 + 1 (space) + 1
	// (newline) = 40 bytes. We'll provide a bit more for good measure.
	//
	// Always read from the beginning of the file to get a fresh value.
	var b [64]byte
	n, errno := linux.Pread(fd, b[:], 0)
	if errno != 0 {
		return 0, false, errSyscallFailed
	}
	if n == len(b) {
		return 0, false, errMalformedFile
	}

	buf := b[:n]
	return parseV2Limit(buf)
}

// FindCPU finds the path to the CPU cgroup that this process is a member of
// and places it in out. scratch is a scratch buffer for internal use.
//
// out must have length PathSize. scratch must have length ParseSize.
//
// Returns the number of bytes written to out and the cgroup version (1 or 2).
//
// Returns ErrNoCgroup if the process is not in a CPU cgroup.
func FindCPU(out []byte, scratch []byte) (int, Version, error) {
	checkBufferSize(out, PathSize)
	checkBufferSize(scratch, ParseSize)

	// The cgroup path is <cgroup mount point> + <relative path>.
	// relative path is the cgroup relative to the mount root.

	n, version, err := FindCPUCgroup(out, scratch)
	if err != nil {
		return 0, 0, err
	}

	n, err = FindCPUMountPoint(out, out[:n], version, scratch)
	return n, version, err
}

// FindCPUCgroup finds the path to the CPU cgroup that this process is a member of
// and places it in out. scratch is a scratch buffer for internal use.
//
// out must have length PathSize. scratch must have length ParseSize.
//
// Returns the number of bytes written to out and the cgroup version (1 or 2).
//
// Returns ErrNoCgroup if the process is not in a CPU cgroup.
func FindCPUCgroup(out []byte, scratch []byte) (int, Version, error) {
	path := []byte("/proc/self/cgroup\x00")
	fd, errno := linux.Open(&path[0], linux.O_RDONLY|linux.O_CLOEXEC, 0)
	if errno == linux.ENOENT {
		return 0, 0, ErrNoCgroup
	} else if errno != 0 {
		return 0, 0, errSyscallFailed
	}

	// The relative path always starts with /, so we can directly append it
	// to the mount point.
	n, version, err := parseCPUCgroup(fd, linux.Read, out[:], scratch)
	if err != nil {
		linux.Close(fd)
		return 0, 0, err
	}

	linux.Close(fd)
	return n, version, nil
}

// FindCPUMountPoint finds the mount point containing the specified cgroup and
// version with cpu controller, and compose the full path to the cgroup in out.
// scratch is a scratch buffer for internal use.
//
// out must have length PathSize, may overlap with cgroup.
// scratch must have length ParseSize.
//
// Returns the number of bytes written to out.
//
// Returns ErrNoCgroup if no matching mount point is found.
func FindCPUMountPoint(out, cgroup []byte, version Version, scratch []byte) (int, error) {
	checkBufferSize(out, PathSize)
	checkBufferSize(scratch, ParseSize)

	path := []byte("/proc/self/mountinfo\x00")
	fd, errno := linux.Open(&path[0], linux.O_RDONLY|linux.O_CLOEXEC, 0)
	if errno == linux.ENOENT {
		return 0, ErrNoCgroup
	} else if errno != 0 {
		return 0, errSyscallFailed
	}

	n, err := parseCPUMount(fd, linux.Read, out, cgroup, version, scratch)
	if err != nil {
		linux.Close(fd)
		return 0, err
	}
	linux.Close(fd)

	return n, nil
}

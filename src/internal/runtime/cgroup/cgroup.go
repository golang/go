// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgroup

import (
	"internal/bytealg"
	"internal/strconv"
)

var (
	ErrNoCgroup error = stringError("not in a cgroup")

	errMalformedFile error = stringError("malformed file")
)

const _PATH_MAX = 4096

const (
	// Required amount of scratch space for CPULimit.
	//
	// TODO(prattmic): This is shockingly large (~70KiB) due to the (very
	// unlikely) combination of extremely long paths consisting mostly
	// escaped characters. The scratch buffer ends up in .bss in package
	// runtime, so it doesn't contribute to binary size and generally won't
	// be faulted in, but it would still be nice to shrink this. A more
	// complex parser that did not need to keep entire lines in memory
	// could get away with much less. Alternatively, we could do a one-off
	// mmap allocation for this buffer, which is only mapped larger if we
	// actually need the extra space.
	ScratchSize = PathSize + ParseSize

	// Required space to store a path of the cgroup in the filesystem.
	PathSize = _PATH_MAX

	// /proc/self/mountinfo path escape sequences are 4 characters long, so
	// a path consisting entirely of escaped characters could be 4 times
	// larger.
	escapedPathMax = 4 * _PATH_MAX

	// Required space to parse /proc/self/mountinfo and /proc/self/cgroup.
	// See findCPUMount and findCPURelativePath.
	ParseSize = 4 * escapedPathMax
)

// Version indicates the cgroup version.
type Version int

const (
	VersionUnknown Version = iota
	V1
	V2
)

func parseV1Number(buf []byte) (int64, error) {
	// Ignore trailing newline.
	i := bytealg.IndexByte(buf, '\n')
	if i < 0 {
		return 0, errMalformedFile
	}
	buf = buf[:i]

	val, err := strconv.ParseInt(string(buf), 10, 64)
	if err != nil {
		return 0, errMalformedFile
	}

	return val, nil
}

func parseV2Limit(buf []byte) (float64, bool, error) {
	i := bytealg.IndexByte(buf, ' ')
	if i < 0 {
		return 0, false, errMalformedFile
	}

	quotaStr := buf[:i]
	if bytealg.Compare(quotaStr, []byte("max")) == 0 {
		// No limit.
		return 0, false, nil
	}

	periodStr := buf[i+1:]
	// Ignore trailing newline, if any.
	i = bytealg.IndexByte(periodStr, '\n')
	if i < 0 {
		return 0, false, errMalformedFile
	}
	periodStr = periodStr[:i]

	quota, err := strconv.ParseInt(string(quotaStr), 10, 64)
	if err != nil {
		return 0, false, errMalformedFile
	}

	period, err := strconv.ParseInt(string(periodStr), 10, 64)
	if err != nil {
		return 0, false, errMalformedFile
	}

	return float64(quota) / float64(period), true, nil
}

// Finds the path of the current process's CPU cgroup relative to the cgroup
// mount and writes it to out.
//
// Returns the number of bytes written and the cgroup version (1 or 2).
func parseCPURelativePath(fd int, read func(fd int, b []byte) (int, uintptr), out []byte, scratch []byte) (int, Version, error) {
	// The format of each line is
	//
	//   hierarchy-ID:controller-list:cgroup-path
	//
	// controller-list is comma-separated.
	// See man 5 cgroup for more details.
	//
	// cgroup v2 has hierarchy-ID 0. If a v1 hierarchy contains "cpu", that
	// is the CPU controller. Otherwise the v2 hierarchy (if any) is the
	// CPU controller.
	//
	// hierarchy-ID and controller-list have relatively small maximum
	// sizes, and the path can be up to _PATH_MAX, so we need a bit more
	// than 1 _PATH_MAX of scratch space.

	l := newLineReader(fd, scratch, read)

	// Bytes written to out.
	n := 0

	for {
		err := l.next()
		if err == errIncompleteLine {
			// Don't allow incomplete lines. While in theory the
			// incomplete line may be for a controller we don't
			// care about, in practice all lines should be of
			// similar length, so we should just have a buffer big
			// enough for any.
			return 0, 0, err
		} else if err == errEOF {
			break
		} else if err != nil {
			return 0, 0, err
		}

		line := l.line()

		// The format of each line is
		//
		//   hierarchy-ID:controller-list:cgroup-path
		//
		// controller-list is comma-separated.
		// See man 5 cgroup for more details.
		i := bytealg.IndexByte(line, ':')
		if i < 0 {
			return 0, 0, errMalformedFile
		}

		hierarchy := line[:i]
		line = line[i+1:]

		i = bytealg.IndexByte(line, ':')
		if i < 0 {
			return 0, 0, errMalformedFile
		}

		controllers := line[:i]
		line = line[i+1:]

		path := line

		if string(hierarchy) == "0" {
			// v2 hierarchy.
			n = copy(out, path)
			// Keep searching, we might find a v1 hierarchy with a
			// CPU controller, which takes precedence.
		} else {
			// v1 hierarchy
			if containsCPU(controllers) {
				// Found a v1 CPU controller. This must be the
				// only one, so we're done.
				return copy(out, path), V1, nil
			}
		}
	}

	if n == 0 {
		// Found nothing.
		return 0, 0, ErrNoCgroup
	}

	// Must be v2, v1 returns above.
	return n, V2, nil
}

// Returns true if comma-separated list b contains "cpu".
func containsCPU(b []byte) bool {
	for len(b) > 0 {
		i := bytealg.IndexByte(b, ',')
		if i < 0 {
			// Neither cmd/compile nor gccgo allocates for these string conversions.
			return string(b) == "cpu"
		}

		curr := b[:i]
		rest := b[i+1:]

		if string(curr) == "cpu" {
			return true
		}

		b = rest
	}

	return false
}

// Returns the mount point for the cpu cgroup controller (v1 or v2) from
// /proc/self/mountinfo.
func parseCPUMount(fd int, read func(fd int, b []byte) (int, uintptr), out []byte, scratch []byte) (int, error) {
	// The format of each line is:
	//
	// 36 35 98:0 /mnt1 /mnt2 rw,noatime master:1 - ext3 /dev/root rw,errors=continue
	// (1)(2)(3)   (4)   (5)      (6)      (7)   (8) (9)   (10)         (11)
	//
	// (1) mount ID:  unique identifier of the mount (may be reused after umount)
	// (2) parent ID:  ID of parent (or of self for the top of the mount tree)
	// (3) major:minor:  value of st_dev for files on filesystem
	// (4) root:  root of the mount within the filesystem
	// (5) mount point:  mount point relative to the process's root
	// (6) mount options:  per mount options
	// (7) optional fields:  zero or more fields of the form "tag[:value]"
	// (8) separator:  marks the end of the optional fields
	// (9) filesystem type:  name of filesystem of the form "type[.subtype]"
	// (10) mount source:  filesystem specific information or "none"
	// (11) super options:  per super block options
	//
	// See man 5 proc_pid_mountinfo for more details.
	//
	// Note that emitted paths will not contain space, tab, newline, or
	// carriage return. Those are escaped. See Linux show_mountinfo ->
	// show_path. We must unescape before returning.
	//
	// We return the mount point (5) if the filesystem type (9) is cgroup2,
	// or cgroup with "cpu" in the super options (11).
	//
	// (4), (5), and (10) are up to _PATH_MAX. The remaining fields have a
	// small fixed maximum size, so 4*_PATH_MAX is plenty of scratch space.
	// Note that non-cgroup mounts may have arbitrarily long (11), but we
	// can skip those when parsing.

	l := newLineReader(fd, scratch, read)

	// Bytes written to out.
	n := 0

	for {
		//incomplete := false
		err := l.next()
		if err == errIncompleteLine {
			// An incomplete line is fine as long as it doesn't
			// impede parsing the fields we need. It shouldn't be
			// possible for any mount to use more than 3*PATH_MAX
			// before (9) because there are two paths and all other
			// earlier fields have bounded options. Only (11) has
			// unbounded options.
		} else if err == errEOF {
			break
		} else if err != nil {
			return 0, err
		}

		line := l.line()

		// Skip first four fields.
		for range 4 {
			i := bytealg.IndexByte(line, ' ')
			if i < 0 {
				return 0, errMalformedFile
			}
			line = line[i+1:]
		}

		// (5) mount point:  mount point relative to the process's root
		i := bytealg.IndexByte(line, ' ')
		if i < 0 {
			return 0, errMalformedFile
		}
		mnt := line[:i]
		line = line[i+1:]

		// Skip ahead past optional fields, delimited by " - ".
		for {
			i = bytealg.IndexByte(line, ' ')
			if i < 0 {
				return 0, errMalformedFile
			}
			if i+3 >= len(line) {
				return 0, errMalformedFile
			}
			delim := line[i : i+3]
			if string(delim) == " - " {
				line = line[i+3:]
				break
			}
			line = line[i+1:]
		}

		// (9) filesystem type:  name of filesystem of the form "type[.subtype]"
		i = bytealg.IndexByte(line, ' ')
		if i < 0 {
			return 0, errMalformedFile
		}
		ftype := line[:i]
		line = line[i+1:]

		if string(ftype) != "cgroup" && string(ftype) != "cgroup2" {
			continue
		}

		// As in findCPUPath, cgroup v1 with a CPU controller takes
		// precendence over cgroup v2.
		if string(ftype) == "cgroup2" {
			// v2 hierarchy.
			n, err = unescapePath(out, mnt)
			if err != nil {
				// Don't keep searching on error. The kernel
				// should never produce broken escaping.
				return n, err
			}
			// Keep searching, we might find a v1 hierarchy with a
			// CPU controller, which takes precedence.
			continue
		}

		// (10) mount source:  filesystem specific information or "none"
		i = bytealg.IndexByte(line, ' ')
		if i < 0 {
			return 0, errMalformedFile
		}
		// Don't care about mount source.
		line = line[i+1:]

		// (11) super options:  per super block options
		superOpt := line

		// v1 hierarchy
		if containsCPU(superOpt) {
			// Found a v1 CPU controller. This must be the
			// only one, so we're done.
			return unescapePath(out, mnt)
		}
	}

	if n == 0 {
		// Found nothing.
		return 0, ErrNoCgroup
	}

	return n, nil
}

var errInvalidEscape error = stringError("invalid path escape sequence")

// unescapePath copies in to out, unescaping escape sequences generated by
// Linux's show_path.
//
// That is, '\', ' ', '\t', and '\n' are converted to octal escape sequences,
// like '\040' for space.
//
// out must be at least as large as in.
//
// Returns the number of bytes written to out.
//
// Also see escapePath in cgroup_linux_test.go.
func unescapePath(out []byte, in []byte) (int, error) {
	// Not strictly necessary, but simplifies the implementation and will
	// always hold in users.
	if len(out) < len(in) {
		throw("output too small")
	}

	var outi, ini int
	for ini < len(in) {
		c := in[ini]
		if c != '\\' {
			out[outi] = c
			outi++
			ini++
			continue
		}

		// Start of escape sequence.

		// Escape sequence is always 4 characters: one slash and three
		// digits.
		if ini+3 >= len(in) {
			return outi, errInvalidEscape
		}

		var outc byte
		for i := range 3 {
			c := in[ini+1+i]
			if c < '0' || c > '9' {
				return outi, errInvalidEscape
			}

			outc *= 8
			outc += c - '0'
		}

		out[outi] = outc
		outi++

		ini += 4
	}

	return outi, nil
}

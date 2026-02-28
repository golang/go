// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux

package pprof

import (
	"fmt"
	"regexp"
	"strconv"
	"syscall"
)

var versionRe = regexp.MustCompile(`^(\d+)(?:\.(\d+)(?:\.(\d+))).*$`)

func linuxKernelVersion() (major, minor, patch int, err error) {
	var uname syscall.Utsname
	if err := syscall.Uname(&uname); err != nil {
		return 0, 0, 0, err
	}

	buf := make([]byte, 0, len(uname.Release))
	for _, b := range uname.Release {
		if b == 0 {
			break
		}
		buf = append(buf, byte(b))
	}
	rl := string(buf)

	m := versionRe.FindStringSubmatch(rl)
	if m == nil {
		return 0, 0, 0, fmt.Errorf("error matching version number in %q", rl)
	}

	v, err := strconv.ParseInt(m[1], 10, 64)
	if err != nil {
		return 0, 0, 0, fmt.Errorf("error parsing major version %q in %s: %w", m[1], rl, err)
	}
	major = int(v)

	if len(m) >= 3 {
		v, err := strconv.ParseInt(m[2], 10, 64)
		if err != nil {
			return 0, 0, 0, fmt.Errorf("error parsing minor version %q in %s: %w", m[2], rl, err)
		}
		minor = int(v)
	}

	if len(m) >= 4 {
		v, err := strconv.ParseInt(m[3], 10, 64)
		if err != nil {
			return 0, 0, 0, fmt.Errorf("error parsing patch version %q in %s: %w", m[3], rl, err)
		}
		patch = int(v)
	}

	return
}

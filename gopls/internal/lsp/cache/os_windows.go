// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
package cache

import (
	"fmt"
	"path/filepath"
	"syscall"
)

func init() {
	checkPathCase = windowsCheckPathCase
}

func windowsCheckPathCase(path string) error {
	// Back in the day, Windows used to have short and long filenames, and
	// it still supports those APIs. GetLongPathName gets the real case for a
	// path, so we can use it here. Inspired by
	// http://stackoverflow.com/q/2113822.

	// Short paths can be longer than long paths, and unicode, so be generous.
	buflen := 4 * len(path)
	namep, err := syscall.UTF16PtrFromString(path)
	if err != nil {
		return err
	}
	short := make([]uint16, buflen)
	n, err := syscall.GetShortPathName(namep, &short[0], uint32(len(short)*2)) // buflen is in bytes.
	if err != nil {
		return err
	}
	if int(n) > len(short)*2 {
		return fmt.Errorf("short buffer too short: %v vs %v*2", n, len(short))
	}
	long := make([]uint16, buflen)
	n, err = syscall.GetLongPathName(&short[0], &long[0], uint32(len(long)*2))
	if err != nil {
		return err
	}
	if int(n) > len(long)*2 {
		return fmt.Errorf("long buffer too short: %v vs %v*2", n, len(long))
	}
	longstr := syscall.UTF16ToString(long)

	isRoot := func(p string) bool {
		return p[len(p)-1] == filepath.Separator
	}
	for got, want := path, longstr; !isRoot(got) && !isRoot(want); got, want = filepath.Dir(got), filepath.Dir(want) {
		if g, w := filepath.Base(got), filepath.Base(want); g != w {
			return fmt.Errorf("case mismatch in path %q: component %q is listed by Windows as %q", path, g, w)
		}
	}
	return nil
}

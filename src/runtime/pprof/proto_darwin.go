// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pprof

import (
	"errors"
)

// readMapping adds a mapping entry for the text region of the running process.
// It uses the mach_vm_region region system call to add mapping entries for the
// text region of the running process. Note that currently no attempt is
// made to obtain the buildID information.
func (b *profileBuilder) readMapping() {
	if !machVMInfo(b.addMapping) {
		b.addMappingEntry(0, 0, 0, "", "", true)
	}
}

func readMainModuleMapping() (start, end uint64, exe, buildID string, err error) {
	first := true
	ok := machVMInfo(func(lo, hi, off uint64, file, build string) {
		if first {
			start, end = lo, hi
			exe, buildID = file, build
		}
		// May see multiple text segments if rosetta is used for running
		// the go toolchain itself.
		first = false
	})
	if !ok {
		return 0, 0, "", "", errors.New("machVMInfo failed")
	}
	return start, end, exe, buildID, nil
}

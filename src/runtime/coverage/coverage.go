// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package coverage

import (
	"internal/coverage/cfile"
	"io"
)

// initHook is invoked from main.init in programs built with -cover.
// The call is emitted by the compiler.
func initHook(istest bool) {
	cfile.InitHook(istest)
}

// WriteMetaDir writes a coverage meta-data file for the currently
// running program to the directory specified in 'dir'. An error will
// be returned if the operation can't be completed successfully (for
// example, if the currently running program was not built with
// "-cover", or if the directory does not exist).
func WriteMetaDir(dir string) error {
	return cfile.WriteMetaDir(dir)
}

// WriteMeta writes the meta-data content (the payload that would
// normally be emitted to a meta-data file) for the currently running
// program to the writer 'w'. An error will be returned if the
// operation can't be completed successfully (for example, if the
// currently running program was not built with "-cover", or if a
// write fails).
func WriteMeta(w io.Writer) error {
	return cfile.WriteMeta(w)
}

// WriteCountersDir writes a coverage counter-data file for the
// currently running program to the directory specified in 'dir'. An
// error will be returned if the operation can't be completed
// successfully (for example, if the currently running program was not
// built with "-cover", or if the directory does not exist). The
// counter data written will be a snapshot taken at the point of the
// call.
func WriteCountersDir(dir string) error {
	return cfile.WriteCountersDir(dir)
}

// WriteCounters writes coverage counter-data content for the
// currently running program to the writer 'w'. An error will be
// returned if the operation can't be completed successfully (for
// example, if the currently running program was not built with
// "-cover", or if a write fails). The counter data written will be a
// snapshot taken at the point of the invocation.
func WriteCounters(w io.Writer) error {
	return cfile.WriteCounters(w)
}

// ClearCounters clears/resets all coverage counter variables in the
// currently running program. It returns an error if the program in
// question was not built with the "-cover" flag. Clearing of coverage
// counters is also not supported for programs not using atomic
// counter mode (see more detailed comments below for the rationale
// here).
func ClearCounters() error {
	return cfile.ClearCounters()
}

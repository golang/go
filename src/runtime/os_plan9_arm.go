// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

func checkgoarm() {
	return // TODO(minux)
}

//go:nosplit
func cputicks() int64 {
	// runtime·nanotime() is a poor approximation of CPU ticks that is enough for the profiler.
	return nanotime()
}

//go:nosplit
func walltime() (sec int64, nsec int32) {
	var t [1]uint64
	readtime(&t[0], 1, 1)
	return timesplit(frombe(t[0]))
}

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !windows

package runtime

// osRelaxMinNS is the number of nanoseconds of idleness to tolerate
// without performing an osRelax. Since osRelax may reduce the
// precision of timers, this should be enough larger than the relaxed
// timer precision to keep the timer error acceptable.
const osRelaxMinNS = 0

var haveHighResSleep = true

// osRelax is called by the scheduler when transitioning to and from
// all Ps being idle.
func osRelax(relax bool) {}

// enableWER is called by setTraceback("wer").
// Windows Error Reporting (WER) is only supported on Windows.
func enableWER() {}

// winlibcall is not implemented on non-Windows systems,
// but it is used in non-OS-specific parts of the runtime.
// Define it as an empty struct to avoid wasting stack space.
type winlibcall struct{}

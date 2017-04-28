// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

var labelSync uintptr

//go:linkname runtime_setProfLabel runtime/pprof.runtime_setProfLabel
func runtime_setProfLabel(labels unsafe.Pointer) {
	// Introduce race edge for read-back via profile.
	// This would more properly use &getg().labels as the sync address,
	// but we do the read in a signal handler and can't call the race runtime then.
	if raceenabled {
		racerelease(unsafe.Pointer(&labelSync))
	}
	getg().labels = labels
}

//go:linkname runtime_getProfLabel runtime/pprof.runtime_getProfLabel
func runtime_getProfLabel() unsafe.Pointer {
	return getg().labels
}

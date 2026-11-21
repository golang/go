// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/runtime/sys"
	_ "unsafe"
)

//go:linkname dit_setEnabled crypto/subtle.setDITEnabled
func dit_setEnabled() bool {
	g := getg()
	g.ditWanted = true
	g.m.ditEnabled = true
	return sys.EnableDIT()
}

//go:linkname dit_setDisabled crypto/subtle.setDITDisabled
func dit_setDisabled() {
	g := getg()
	g.ditWanted = false
	g.m.ditEnabled = false
	sys.DisableDIT()
}

// syncDIT enables or disables DIT on mp, the current M, as wanted by gp,
// which is about to run on it.
func syncDIT(gp *g, mp *m) {
	if sys.DITSupported && debug.dataindependenttiming != 1 {
		if gp.ditWanted && !mp.ditEnabled {
			// The current M doesn't have DIT enabled, but the goroutine we're
			// executing does need it, so turn it on.
			sys.EnableDIT()
			mp.ditEnabled = true
		} else if !gp.ditWanted && mp.ditEnabled {
			// The current M has DIT enabled, but the goroutine we're executing does
			// not need it, so turn it off.
			// NOTE: turning off DIT here means that the scheduler will have DIT enabled
			// when it runs after this goroutine yields or is preempted. This may have
			// a minor performance impact on the scheduler.
			sys.DisableDIT()
			mp.ditEnabled = false
		}
	}
}

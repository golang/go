// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import "sync/atomic"

// Upon calling ResetId, nextId starts with 1 again.
// It may be called concurrently. This is only needed
// for tests where we may want to have a consistent
// numbering for each individual test case.
func ResetId() { atomic.StoreUint32(&lastId, 0) }

// SetGoVersion sets the unexported goVersion field on config, so that tests
// which assert on behavior for older Go versions can set it.
func SetGoVersion(config *Config, goVersion string) {
	config.goVersion = goVersion
}

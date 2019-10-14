// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build cmd_go_bootstrap

package modfetch

import "cmd/go/internal/module"

func useSumDB(mod module.Version) bool {
	return false
}

func lookupSumDB(mod module.Version) (string, []string, error) {
	panic("bootstrap")
}

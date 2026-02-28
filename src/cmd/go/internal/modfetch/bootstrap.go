// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build cmd_go_bootstrap

package modfetch

import "golang.org/x/mod/module"

func useSumDB(mod module.Version) bool {
	return false
}

func lookupSumDB(mod module.Version) (string, []string, error) {
	panic("bootstrap")
}

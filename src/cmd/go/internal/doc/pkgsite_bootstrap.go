// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build cmd_go_bootstrap

// Don't build the pkgsite code into go_bootstrap because it depends on net.

package doc

func doPkgsite(string) error { return nil }

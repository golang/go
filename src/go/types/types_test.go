// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

// SetGoVersion sets the unexported goVersion field on config, so that tests
// which assert on behavior for older Go versions can set it.
func SetGoVersion(config *Config, goVersion string) {
	config.goVersion = goVersion
}

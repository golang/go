// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package work

import (
	"path/filepath"

	"cmd/go/internal/cfg"
	"cmd/internal/pathcache"
)

func findWasmExecCmd(execCmd []string) []string {
	if len(execCmd) != 0 || cfg.GOROOT == "" || cfg.Goarch != "wasm" || (cfg.Goos != "js" && cfg.Goos != "wasip1") {
		return execCmd
	}
	path, err := pathcache.LookPath(filepath.Join(cfg.GOROOT, "lib", "wasm", "go_"+cfg.Goos+"_wasm_exec"))
	if err != nil {
		return execCmd
	}
	return []string{path}
}

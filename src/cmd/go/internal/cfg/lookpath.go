// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cfg

import (
	"cmd/internal/par"
	"os/exec"
)

var lookPathCache par.ErrCache[string, string]

// LookPath wraps exec.LookPath and caches the result
// which can be called by multiple Goroutines at the same time.
func LookPath(file string) (path string, err error) {
	return lookPathCache.Do(file,
		func() (string, error) {
			return exec.LookPath(file)
		})
}

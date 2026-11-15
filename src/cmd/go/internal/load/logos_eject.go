// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package load

import (
	"path/filepath"
)

var logosEjectDir string

func ArmLogosEject(dir string) {
	logosEjectDir = dir
}

func DisarmLogosEject() {
	logosEjectDir = ""
}

func logosEjectGcflags(p *Package) []string {
	if logosEjectDir == "" || p.Standard || !p.IsLogos() {
		return nil
	}
	p.Internal.LogosEject = true
	return []string{"-logos.eject=" + filepath.Join(logosEjectDir, filepath.FromSlash(p.ImportPath))}
}

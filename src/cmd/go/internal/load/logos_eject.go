// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package load

import (
	"path/filepath"
)

// This file holds the cmd/go side of logos eject (M1-S6): marking the
// packages whose compile actions must emit lowered Go source, and the
// per-package gcflags hook that passes the emission directory to the
// compiler flag -logos.eject=DIR (provided by the compiler side).
//
// The eject command arms emission with ArmLogosEject before loading
// packages; setToolFlags in pkg.go then marks every non-standard loGo/s
// package and appends its flag through the single hook line there.
// Plain go and ordinary logos builds never arm, so the hook returns nil
// and their action IDs stay unchanged.

// logosEjectDir stages compiler-emitted Go source during logos eject.
// Empty when no eject is running.
var logosEjectDir string

// ArmLogosEject stages compiler-emitted Go source under dir for the next
// package load: every non-standard loGo/s package loaded while armed is
// marked for emission and gets -logos.eject=<dir>/<import path> in its
// per-package gcflags.
func ArmLogosEject(dir string) {
	logosEjectDir = dir
}

// DisarmLogosEject stops emission marking for later loads in this process.
func DisarmLogosEject() {
	logosEjectDir = ""
}

// logosEjectGcflags is the setToolFlags hook for logos eject. While armed,
// it marks p for emission and returns the -logos.eject flag directing the
// compiler to write the package's lowered files under the staging
// directory; otherwise it returns nil and changes nothing. It touches no
// filesystem state: the eject command creates the staging directories
// after loading and before building.
func logosEjectGcflags(p *Package) []string {
	if logosEjectDir == "" || p.Standard || !p.IsLogos() {
		return nil
	}
	p.Internal.LogosEject = true
	return []string{"-logos.eject=" + filepath.Join(logosEjectDir, filepath.FromSlash(p.ImportPath))}
}

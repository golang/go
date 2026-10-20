// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package load

import (
	"path/filepath"
	"strings"
)

// logosNextOut stages the compiler sidecar during logos next. Empty
// when no next scan is running.
var logosNextOut string

// ArmLogosNext stages compiler scan output at out for the next package
// load: every main-module Go package loaded while armed is marked for
// scanning and gets -logos.mode=next with -logos.nextout=out in its
// per-package gcflags.
func ArmLogosNext(out string) {
	logosNextOut = out
}

// DisarmLogosNext stops scan marking for later loads in this process.
func DisarmLogosNext() {
	logosNextOut = ""
}

// HasNextFiles reports whether p holds Go files worth scanning.
func HasNextFiles(p *Package) bool {
	for _, files := range [][]string{p.GoFiles, p.TestGoFiles, p.XTestGoFiles, p.CgoFiles} {
		for _, name := range files {
			if strings.HasSuffix(name, ".go") {
				return true
			}
		}
	}
	return false
}

// logosNextGcflags is the setToolFlags hook for logos next. While armed,
// it marks p for scanning and returns the mode and sidecar flags
// directing the compiler to append the package findings; otherwise it
// returns nil and changes nothing.
func logosNextGcflags(p *Package) []string {
	if logosNextOut == "" || p.Standard || p.IsLogos() {
		return nil
	}
	if p.Internal.TestmainGo != nil {
		return nil
	}
	if p.Module == nil || !p.Module.Main || !HasNextFiles(p) {
		return nil
	}
	p.Internal.LogosNext = true
	return []string{"-logos.mode=next", "-logos.nextout=" + filepath.Clean(logosNextOut)}
}

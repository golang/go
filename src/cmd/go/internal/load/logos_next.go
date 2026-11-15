// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package load

import (
	"path/filepath"
	"strings"
)

var logosNextOut string

func ArmLogosNext(out string) {
	logosNextOut = out
}

func DisarmLogosNext() {
	logosNextOut = ""
}

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

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// installLogos copies the just installed go binary to logos in the same
// directory, using the suffix for the current target.
func installLogos() {
	dir := gorootBin
	suffix := exe
	if goos != gohostos || goarch != gohostarch {
		dir = pathf("%s/bin/%s_%s", goroot, goos, goarch)
		suffix = ""
		if goos == "windows" {
			suffix = ".exe"
		}
	}
	copyfile(pathf("%s/logos"+suffix, dir), pathf("%s/go"+suffix, dir), writeExec)
}

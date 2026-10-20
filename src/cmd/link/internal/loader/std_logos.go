// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package loader

// StdPackages returns the import paths of the loaded packages the compiler
// built as standard library: cmd/go passes -std for them and the object
// file records the flag, so the answer does not depend on GOROOT, on
// -trimpath, or on the shape of the import path.
func (l *Loader) StdPackages() map[string]bool {
	std := make(map[string]bool)
	for _, r := range l.objs {
		if r != nil && r != l.extReader && r.Std() {
			std[r.unit.Lib.Pkg] = true
		}
	}
	return std
}

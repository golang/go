// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package loader

func (l *Loader) StdPackages() map[string]bool {
	std := make(map[string]bool)
	for _, r := range l.objs {
		if r != nil && r != l.extReader && r.Std() {
			std[r.unit.Lib.Pkg] = true
		}
	}
	return std
}

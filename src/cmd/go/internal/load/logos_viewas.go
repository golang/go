// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package load

import "cmd/go/internal/cfg"

// logosSkipVirtualImport reports whether path is a virtual import with
// no files to load: "comptime" is synthesized by the type checker in
// logos mode (M7-S5), so package loading drops it like "C" instead of
// resolving it to a directory. The M7-S2 comptime slice owns the wider
// load story; this skip is the minimal cut that lets logos build reach
// the compiler for ViewAs-only files.
func logosSkipVirtualImport(path string) bool {
	return cfg.Logos && path == "comptime"
}

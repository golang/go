// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package load

import "cmd/go/internal/cfg"

func logosSkipVirtualImport(path string) bool {
	return cfg.Logos && path == "comptime"
}

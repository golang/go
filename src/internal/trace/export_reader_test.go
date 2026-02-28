// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trace

import "internal/trace/version"

// GoVersion is the version set in the trace header.
func (r *Reader) GoVersion() version.Version {
	return r.version
}

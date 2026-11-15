// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package objabi

import "strings"

func StripExperiment(version string) string {
	if before, _, ok := strings.Cut(version, " X:"); ok {
		return before
	}
	if before, _, ok := strings.Cut(version, "-X:"); ok {
		return before
	}
	return version
}

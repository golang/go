// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package work

import (
	"slices"
	"strings"
)

func logosReportsBuildID(f []string) bool {
	return strings.HasPrefix(f[len(f)-1], "buildID=")
}

func logosEjectNoCache(a *Action) bool {
	if a == nil || a.Package == nil {
		return false
	}
	return a.Package.Internal.LogosEject || slices.ContainsFunc(a.Package.Internal.Gcflags, logosEjectFlag)
}

func logosEjectFlag(f string) bool {
	return strings.HasPrefix(f, "-logos.eject=")
}

func logosNextNoCache(a *Action) bool {
	return a != nil && a.Package != nil && a.Package.Internal.LogosNext
}

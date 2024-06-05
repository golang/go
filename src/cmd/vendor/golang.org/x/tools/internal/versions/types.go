// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package versions

import (
	"go/types"
)

// GoVersion returns the Go version of the type package.
// It returns zero if no version can be determined.
func GoVersion(pkg *types.Package) string {
	// TODO(taking): x/tools can call GoVersion() [from 1.21] after 1.25.
	if pkg, ok := any(pkg).(interface{ GoVersion() string }); ok {
		return pkg.GoVersion()
	}
	return ""
}

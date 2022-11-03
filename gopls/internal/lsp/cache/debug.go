// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"fmt"
	"os"
	"sort"
)

// This file contains helpers that can be used to instrument code while
// debugging.

// debugEnabled toggles the helpers below.
const debugEnabled = false

// If debugEnabled is true, debugf formats its arguments and prints to stderr.
// If debugEnabled is false, it is a no-op.
func debugf(format string, args ...interface{}) {
	if !debugEnabled {
		return
	}
	if false {
		_ = fmt.Sprintf(format, args...) // encourage vet to validate format strings
	}
	fmt.Fprintf(os.Stderr, ">>> "+format+"\n", args...)
}

// If debugEnabled is true, dumpWorkspace prints a summary of workspace
// packages to stderr. If debugEnabled is false, it is a no-op.
func (s *snapshot) dumpWorkspace(context string) {
	if !debugEnabled {
		return
	}

	debugf("workspace (after %s):", context)
	var ids []PackageID
	for id := range s.workspacePackages {
		ids = append(ids, id)
	}

	sort.Slice(ids, func(i, j int) bool {
		return ids[i] < ids[j]
	})

	for _, id := range ids {
		pkgPath := s.workspacePackages[id]
		_, ok := s.meta.metadata[id]
		debugf("  %s:%s (metadata: %t)", id, pkgPath, ok)
	}
}

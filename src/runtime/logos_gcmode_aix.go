// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build aix

package runtime

// logosPhysicalMemory returns 0: the runtime has no physical memory source on
// AIX, so batch there needs GOMEMLIMIT to trigger a collection.
func logosPhysicalMemory() uint64 { return 0 }

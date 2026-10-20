// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package objabi

// forkReportsBuildID makes -V=full print the build ID of the tool binary
// even though the loGos toolchain reports the release version go1.27.2.
// The go command keys the build cache on the -V=full line. With only the
// release version, a rebuilt compiler, assembler, or linker shares cache
// keys with the earlier build, so the go command reuses archives that the
// earlier binary wrote. The fork changes these tools under one release
// version, so every tool reports its build ID, as development builds do.
const forkReportsBuildID = true

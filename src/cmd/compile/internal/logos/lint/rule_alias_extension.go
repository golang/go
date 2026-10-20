// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

// aliasExtension is portability::alias_extension (SPEC Sec 1.2, T-101.1): a
// source file with a loGo/s alias extension in a main module whose path starts
// with a dotted element, a module that could be published, where the module
// zip rules reject the file name. The go command already reports it under
// this ID while it loads the package (cmd/go/internal/load, M0-S2), as a
// warning, before the compiler runs. The entry has no Check, because the
// compiler sees only the source text, not the file extension rules of the
// module, and it carries the group and scope. The rule never applies to Go
// packages.
var aliasExtension = Rule{
	ID:    "portability::alias_extension",
	Group: Portability,
}

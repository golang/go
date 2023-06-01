// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gover

import "golang.org/x/mod/modfile"

const (
	// narrowAllVersion is the Go version at which the
	// module-module "all" pattern no longer closes over the dependencies of
	// tests outside of the main module.
	NarrowAllVersion = "1.16"

	// DefaultGoModVersion is the Go version to assume for go.mod files
	// that do not declare a Go version. The go command has been
	// writing go versions to modules since Go 1.12, so a go.mod
	// without a version is either very old or recently hand-written.
	// Since we can't tell which, we have to assume it's very old.
	// The semantics of the go.mod changed at Go 1.17 to support
	// graph pruning. If see a go.mod without a go line, we have to
	// assume Go 1.16 so that we interpret the requirements correctly.
	// Note that this default must stay at Go 1.16; it cannot be moved forward.
	DefaultGoModVersion = "1.16"

	// DefaultGoWorkVersion is the Go version to assume for go.work files
	// that do not declare a Go version. Workspaces were added in Go 1.18,
	// so use that.
	DefaultGoWorkVersion = "1.18"

	// ExplicitIndirectVersion is the Go version at which a
	// module's go.mod file is expected to list explicit requirements on every
	// module that provides any package transitively imported by that module.
	//
	// Other indirect dependencies of such a module can be safely pruned out of
	// the module graph; see https://golang.org/ref/mod#graph-pruning.
	ExplicitIndirectVersion = "1.17"

	// separateIndirectVersion is the Go version at which
	// "// indirect" dependencies are added in a block separate from the direct
	// ones. See https://golang.org/issue/45965.
	SeparateIndirectVersion = "1.17"

	// tidyGoModSumVersion is the Go version at which
	// 'go mod tidy' preserves go.mod checksums needed to build test dependencies
	// of packages in "all", so that 'go test all' can be run without checksum
	// errors.
	// See https://go.dev/issue/56222.
	TidyGoModSumVersion = "1.21"

	// goStrictVersion is the Go version at which the Go versions
	// became "strict" in the sense that, restricted to modules at this version
	// or later, every module must have a go version line ≥ all its dependencies.
	// It is also the version after which "too new" a version is considered a fatal error.
	GoStrictVersion = "1.21"
)

// FromGoMod returns the go version from the go.mod file.
// It returns DefaultGoModVersion if the go.mod file does not contain a go line or if mf is nil.
func FromGoMod(mf *modfile.File) string {
	if mf == nil || mf.Go == nil {
		return DefaultGoModVersion
	}
	return mf.Go.Version
}

// FromGoWork returns the go version from the go.mod file.
// It returns DefaultGoWorkVersion if the go.mod file does not contain a go line or if wf is nil.
func FromGoWork(wf *modfile.WorkFile) string {
	if wf == nil || wf.Go == nil {
		return DefaultGoWorkVersion
	}
	return wf.Go.Version
}

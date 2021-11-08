// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mvs

import (
	"fmt"
	"strings"

	"golang.org/x/mod/module"
)

// BuildListError decorates an error that occurred gathering requirements
// while constructing a build list. BuildListError prints the chain
// of requirements to the module where the error occurred.
type BuildListError struct {
	Err   error
	stack []buildListErrorElem
}

type buildListErrorElem struct {
	m module.Version

	// nextReason is the reason this module depends on the next module in the
	// stack. Typically either "requires", or "updating to".
	nextReason string
}

// NewBuildListError returns a new BuildListError wrapping an error that
// occurred at a module found along the given path of requirements and/or
// upgrades, which must be non-empty.
//
// The isVersionChange function reports whether a path step is due to an
// explicit upgrade or downgrade (as opposed to an existing requirement in a
// go.mod file). A nil isVersionChange function indicates that none of the path
// steps are due to explicit version changes.
func NewBuildListError(err error, path []module.Version, isVersionChange func(from, to module.Version) bool) *BuildListError {
	stack := make([]buildListErrorElem, 0, len(path))
	for len(path) > 1 {
		reason := "requires"
		if isVersionChange != nil && isVersionChange(path[0], path[1]) {
			reason = "updating to"
		}
		stack = append(stack, buildListErrorElem{
			m:          path[0],
			nextReason: reason,
		})
		path = path[1:]
	}
	stack = append(stack, buildListErrorElem{m: path[0]})

	return &BuildListError{
		Err:   err,
		stack: stack,
	}
}

// Module returns the module where the error occurred. If the module stack
// is empty, this returns a zero value.
func (e *BuildListError) Module() module.Version {
	if len(e.stack) == 0 {
		return module.Version{}
	}
	return e.stack[len(e.stack)-1].m
}

func (e *BuildListError) Error() string {
	b := &strings.Builder{}
	stack := e.stack

	// Don't print modules at the beginning of the chain without a
	// version. These always seem to be the main module or a
	// synthetic module ("target@").
	for len(stack) > 0 && stack[0].m.Version == "" {
		stack = stack[1:]
	}

	if len(stack) == 0 {
		b.WriteString(e.Err.Error())
	} else {
		for _, elem := range stack[:len(stack)-1] {
			fmt.Fprintf(b, "%s %s\n\t", elem.m, elem.nextReason)
		}
		// Ensure that the final module path and version are included as part of the
		// error message.
		m := stack[len(stack)-1].m
		if mErr, ok := e.Err.(*module.ModuleError); ok {
			actual := module.Version{Path: mErr.Path, Version: mErr.Version}
			if v, ok := mErr.Err.(*module.InvalidVersionError); ok {
				actual.Version = v.Version
			}
			if actual == m {
				fmt.Fprintf(b, "%v", e.Err)
			} else {
				fmt.Fprintf(b, "%s (replaced by %s): %v", m, actual, mErr.Err)
			}
		} else {
			fmt.Fprintf(b, "%v", module.VersionError(m, e.Err))
		}
	}
	return b.String()
}

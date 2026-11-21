// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"go/token"
	"regexp"

	"simd/archsimd/_gen/sgutil"
	"simd/archsimd/_gen/specgen"
)

// apiPhase returns the phase of the current target's API files, other than
// the files of experimental ops. SVE is entirely experimental.
func apiPhase() sgutil.Phase {
	if CurrentArch().isSVE() {
		return sgutil.PhaseExperiment
	}
	return sgutil.PhaseStable
}

// opPhase returns the phase of op's Go declaration.
//
// Exported declarations are in [sgutil.PhaseStable] unless spec says
// they're experimental. Unexported declarations are always in
// [sgutil.PhaseStable], since they may implement generally available APIs.
// Ops with no Go declaration are in [sgutil.PhaseExperiment], since no API
// can depend on them.
func opPhase(op Operation, specIdx *specgen.Index) sgutil.Phase {
	recv, name, ok := opDecl(op)
	if !ok {
		return sgutil.PhaseExperiment
	}
	if token.IsExported(name) && specIdx.IsExperiment(recv, name) {
		return sgutil.PhaseExperiment
	}
	return sgutil.PhaseStable
}

var stubDeclRE = regexp.MustCompile(`(?m)^func \(\w+ (\w+)\) (\w+)\(`)

// opDecl returns the receiver type and method name of op's Go declaration, or
// false if op has no Go declaration.
func opDecl(op Operation) (recv, name string, ok bool) {
	stub, ok := opStub(op)
	if !ok {
		return "", "", false
	}
	// The stub templates choose the receiver from among op's operands in ways
	// that vary by template (see stubTemplates), so rather than duplicating
	// that logic, find the receiver and name in the generated declaration.
	m := stubDeclRE.FindSubmatch(stub)
	if m == nil {
		panic(fmt.Errorf("can't find method declaration in stub for op %v:\n%s", op.Go, stub))
	}
	return string(m[1]), string(m[2]), true
}

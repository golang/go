// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"go/token"
	"regexp"

	"simd/archsimd/_gen/specgen"
)

// A phase identifies the GOEXPERIMENT that enables an API. Phases are ordered
// from least to most stable, and the zero phase means the phase is unknown.
type phase int

const (
	// phaseExperiment APIs are enabled only by GOEXPERIMENT=simd.
	phaseExperiment phase = 1

	// phaseStable APIs are generally available, enabled by
	// GOEXPERIMENT=simdstable (which GOEXPERIMENT=simd implies).
	phaseStable phase = 2
)

// experiment returns the GOEXPERIMENT that enables phase p.
func (p phase) experiment() string {
	switch p {
	case phaseStable:
		return "simdstable"
	case phaseExperiment:
		return "simd"
	}
	panic(fmt.Sprintf("unknown phase %d", int(p)))
}

func (p phase) String() string {
	return p.experiment()
}

// opPhase returns the phase of op's Go declaration.
//
// Exported declarations are in phaseStable unless spec says they're
// experimental. Unexported declarations are always in phaseStable, since they
// may implement generally available APIs. Ops with no Go declaration are in
// phaseExperiment, since no API can depend on them.
func opPhase(op Operation, specIdx *specgen.Index) phase {
	recv, name, ok := opDecl(op)
	if !ok {
		return phaseExperiment
	}
	if token.IsExported(name) && specIdx.IsExperiment(recv, name) {
		return phaseExperiment
	}
	return phaseStable
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

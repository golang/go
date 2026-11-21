// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sgutil

import "fmt"

// A Phase identifies the GOEXPERIMENT that enables an API or a file. Phases
// are ordered from least to most stable, and the zero Phase means the phase
// is unknown (or, for an [Output], that it isn't enabled by a GOEXPERIMENT).
type Phase int

const (
	// PhaseExperiment APIs are enabled only by GOEXPERIMENT=simd.
	PhaseExperiment Phase = 1

	// PhaseStable APIs are generally available, enabled by
	// GOEXPERIMENT=simdstable (which GOEXPERIMENT=simd implies).
	PhaseStable Phase = 2
)

// Experiment returns the GOEXPERIMENT that enables phase p.
func (p Phase) Experiment() string {
	switch p {
	case PhaseStable:
		return "simdstable"
	case PhaseExperiment:
		return "simd"
	}
	panic(fmt.Sprintf("unknown phase %d", int(p)))
}

func (p Phase) String() string {
	return p.Experiment()
}

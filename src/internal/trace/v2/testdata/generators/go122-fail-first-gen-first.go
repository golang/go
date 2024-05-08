// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Regression test for #55160.
//
// The issue is that the parser reads ahead to the first batch of the
// next generation to find generation boundaries, but if it finds an
// error, it needs to delay handling that error until later. Previously
// it would handle that error immediately and a totally valid generation
// would be skipped for parsing and rejected because of an error in a
// batch in the following generation.
//
// This test captures this behavior by making both the first generation
// and second generation bad. It requires that the issue in the first
// generation, which is caught when actually ordering events, be reported
// instead of the second one.

package main

import (
	"internal/trace/v2/event/go122"
	testgen "internal/trace/v2/internal/testgen/go122"
)

func main() {
	testgen.Main(gen)
}

func gen(t *testgen.Trace) {
	// A running goroutine emits a task begin.
	t.RawEvent(go122.EvEventBatch, nil, 1 /*gen*/, 0 /*thread ID*/, 0 /*timestamp*/, 5 /*batch length*/)
	t.RawEvent(go122.EvFrequency, nil, 15625000)

	// A running goroutine emits a task begin.
	t.RawEvent(go122.EvEventBatch, nil, 1 /*gen*/, 0 /*thread ID*/, 0 /*timestamp*/, 5 /*batch length*/)
	t.RawEvent(go122.EvGoCreate, nil, 0 /*timestamp delta*/, 1 /*go ID*/, 0, 0)

	// Write an invalid batch event for the next generation.
	t.RawEvent(go122.EvEventBatch, nil, 2 /*gen*/, 0 /*thread ID*/, 0 /*timestamp*/, 50 /*batch length (invalid)*/)

	// We should fail at the first issue, not the second one.
	t.ExpectFailure("expected a proc but didn't have one")
}

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !amd64 && !arm64 && !loong64

// This provides common support for architectures that DO NOT use extended
// register state in asynchronous preemption.

package runtime

type xRegPerG struct{}

type xRegPerP struct{}

// xRegState is defined only so the build fails if we try to define a real
// xRegState on a noxreg architecture.
type xRegState struct{}

func xRegInitAlloc() {}

func xRegSave(gp *g) {}

//go:nosplit
func xRegRestore(gp *g) {}

func (*xRegPerP) free() {}

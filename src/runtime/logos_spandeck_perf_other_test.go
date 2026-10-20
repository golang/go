// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !(linux && (amd64 || arm64))

package runtime_test

import "errors"

// l1Counter has no hardware counter on this platform.
type l1Counter struct{}

func openL1Misses() (*l1Counter, error) { return nil, errors.ErrUnsupported }

func (c *l1Counter) start() {}

func (c *l1Counter) stop() (uint64, bool) { return 0, false }

func (c *l1Counter) close() {}

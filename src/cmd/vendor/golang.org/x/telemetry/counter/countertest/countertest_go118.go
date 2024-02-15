// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !go1.19

package countertest

import "golang.org/x/telemetry/counter"

func Open(telemetryDir string) {}

func ReadCounter(c *counter.Counter) (count uint64, _ error) {
	return 0, nil
}

func ReadStackCounter(c *counter.StackCounter) (stackCounts map[string]uint64, _ error) {
	return nil, nil
}

func ReadFile(name string) (map[string]uint64, map[string]uint64, error) { return nil, nil, nil }

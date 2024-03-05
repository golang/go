// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build windows

package net

import (
	"testing"
	"time"
)

var testConfigs = []KeepAliveConfig{
	{
		Enable:   true,
		Idle:     2 * time.Second,
		Interval: time.Second,
		Count:    -1,
	},
}

func getCurrentKeepAliveSettings(_ int) (cfg KeepAliveConfig, err error) {
	// TODO(panjf2000): same as verifyKeepAliveSettings.
	return
}

func verifyKeepAliveSettings(_ *testing.T, _ int, _, _ KeepAliveConfig) {
	// TODO(panjf2000): Unlike Unix-like OS's, Windows doesn't provide
	// 	any ways to retrieve the current TCP keep-alive settings, therefore
	// 	we're not able to run the test suite similar to Unix-like OS's on Windows.
	//  Try to find another proper approach to test the keep-alive settings on Windows.
}

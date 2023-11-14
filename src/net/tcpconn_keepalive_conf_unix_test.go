// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build aix || freebsd || linux || netbsd || darwin || dragonfly

package net

import "time"

var testConfigs = []KeepAliveConfig{
	{
		Enable:   true,
		Idle:     5 * time.Second,
		Interval: 3 * time.Second,
		Count:    10,
	},
	{
		Enable:   true,
		Idle:     0,
		Interval: 0,
		Count:    0,
	},
	{
		Enable:   true,
		Idle:     -1,
		Interval: -1,
		Count:    -1,
	},
	{
		Enable:   true,
		Idle:     -1,
		Interval: 3 * time.Second,
		Count:    10,
	},
	{
		Enable:   true,
		Idle:     5 * time.Second,
		Interval: -1,
		Count:    10,
	},
	{
		Enable:   true,
		Idle:     5 * time.Second,
		Interval: 3 * time.Second,
		Count:    -1,
	},
	{
		Enable:   true,
		Idle:     -1,
		Interval: -1,
		Count:    10,
	},
	{
		Enable:   true,
		Idle:     -1,
		Interval: 3 * time.Second,
		Count:    -1,
	},
	{
		Enable:   true,
		Idle:     5 * time.Second,
		Interval: -1,
		Count:    -1,
	},
	{
		Enable:   true,
		Idle:     0,
		Interval: 3 * time.Second,
		Count:    10,
	},
	{
		Enable:   true,
		Idle:     5 * time.Second,
		Interval: 0,
		Count:    10,
	},
	{
		Enable:   true,
		Idle:     5 * time.Second,
		Interval: 3 * time.Second,
		Count:    0,
	},
	{
		Enable:   true,
		Idle:     0,
		Interval: 0,
		Count:    10,
	},
	{
		Enable:   true,
		Idle:     0,
		Interval: 3 * time.Second,
		Count:    0,
	},
	{
		Enable:   true,
		Idle:     5 * time.Second,
		Interval: 0,
		Count:    0,
	},
}

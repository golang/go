// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build solaris && !illumos

package net

import (
	"testing"
	"time"
)

const (
	syscall_TCP_KEEPIDLE  = sysTCP_KEEPIDLE
	syscall_TCP_KEEPCNT   = sysTCP_KEEPCNT
	syscall_TCP_KEEPINTVL = sysTCP_KEEPINTVL
)

type fdType = int

func maybeSkipKeepAliveTest(_ *testing.T) {}

var testConfigs = []KeepAliveConfig{
	{
		Enable:   true,
		Idle:     20 * time.Second, // the minimum value is ten seconds on Solaris
		Interval: 10 * time.Second, // ditto
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
		Interval: 10 * time.Second,
		Count:    10,
	},
	{
		Enable:   true,
		Idle:     20 * time.Second,
		Interval: -1,
		Count:    10,
	},
	{
		Enable:   true,
		Idle:     20 * time.Second,
		Interval: 10 * time.Second,
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
		Interval: 10 * time.Second,
		Count:    -1,
	},
	{
		Enable:   true,
		Idle:     20 * time.Second,
		Interval: -1,
		Count:    -1,
	},
	{
		Enable:   true,
		Idle:     0,
		Interval: 10 * time.Second,
		Count:    10,
	},
	{
		Enable:   true,
		Idle:     20 * time.Second,
		Interval: 0,
		Count:    10,
	},
	{
		Enable:   true,
		Idle:     20 * time.Second,
		Interval: 10 * time.Second,
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
		Interval: 10 * time.Second,
		Count:    0,
	},
	{
		Enable:   true,
		Idle:     20 * time.Second,
		Interval: 0,
		Count:    0,
	},
}

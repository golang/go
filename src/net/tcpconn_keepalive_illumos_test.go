// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build illumos

package net

import (
	"syscall"
	"testing"
	"time"
)

func getCurrentKeepAliveSettings(fd fdType) (cfg KeepAliveConfig, err error) {
	tcpKeepAlive, err := syscall.GetsockoptInt(fd, syscall.SOL_SOCKET, syscall.SO_KEEPALIVE)
	if err != nil {
		return
	}
	tcpKeepAliveIdle, err := syscall.GetsockoptInt(fd, syscall.IPPROTO_TCP, syscall_TCP_KEEPIDLE)
	if err != nil {
		return
	}
	tcpKeepAliveInterval, err := syscall.GetsockoptInt(fd, syscall.IPPROTO_TCP, syscall_TCP_KEEPINTVL)
	if err != nil {
		return
	}
	tcpKeepAliveCount, err := syscall.GetsockoptInt(fd, syscall.IPPROTO_TCP, syscall_TCP_KEEPCNT)
	if err != nil {
		return
	}
	cfg = KeepAliveConfig{
		Enable:   tcpKeepAlive != 0,
		Idle:     time.Duration(tcpKeepAliveIdle) * time.Second,
		Interval: time.Duration(tcpKeepAliveInterval) * time.Second,
		Count:    tcpKeepAliveCount,
	}
	return
}

func verifyKeepAliveSettings(t *testing.T, fd fdType, oldCfg, cfg KeepAliveConfig) {
	const defaultTcpKeepAliveAbortThreshold = 8 * time.Minute // default value on illumos

	if cfg.Idle == 0 {
		cfg.Idle = defaultTCPKeepAliveIdle
	}
	if cfg.Interval == 0 {
		cfg.Interval = defaultTCPKeepAliveInterval
	}
	if cfg.Count == 0 {
		cfg.Count = defaultTCPKeepAliveCount
	}

	if cfg.Idle == -1 {
		cfg.Idle = oldCfg.Idle
	}
	// Check out the comment on KeepAliveConfig and the illumos code:
	// https://github.com/illumos/illumos-gate/blob/0886dcadf4b2cd677c3b944167f0d16ccb243616/usr/src/uts/common/inet/tcp/tcp_opt_data.c#L786-L861
	tcpKeepAliveAbortThreshold := defaultTcpKeepAliveAbortThreshold
	switch {
	case cfg.Interval == -1 && cfg.Count == -1:
		cfg.Interval = oldCfg.Interval
		cfg.Count = oldCfg.Count
	case cfg.Interval == -1 && cfg.Count > 0:
		cfg.Interval = defaultTcpKeepAliveAbortThreshold / time.Duration(cfg.Count)
	case cfg.Count == -1 && cfg.Interval > 0:
		cfg.Count = int(defaultTcpKeepAliveAbortThreshold / cfg.Interval)
	case cfg.Interval > 0 && cfg.Count > 0:
		// TCP_KEEPALIVE_ABORT_THRESHOLD will be recalculated only when both TCP_KEEPINTVL
		// and TCP_KEEPCNT are set, otherwise it will remain the default value.
		tcpKeepAliveAbortThreshold = cfg.Interval * time.Duration(cfg.Count)
	}

	tcpKeepAlive, err := syscall.GetsockoptInt(fd, syscall.SOL_SOCKET, syscall.SO_KEEPALIVE)
	if err != nil {
		t.Fatal(err)
	}
	if (tcpKeepAlive != 0) != cfg.Enable {
		t.Fatalf("SO_KEEPALIVE: got %t; want %t", tcpKeepAlive != 0, cfg.Enable)
	}

	tcpKeepAliveIdle, err := syscall.GetsockoptInt(fd, syscall.IPPROTO_TCP, syscall_TCP_KEEPIDLE)
	if err != nil {
		t.Fatal(err)
	}
	if time.Duration(tcpKeepAliveIdle)*time.Second != cfg.Idle {
		t.Fatalf("TCP_KEEPIDLE: got %ds; want %v", tcpKeepAliveIdle, cfg.Idle)
	}
	tcpKeepAliveThreshold, err := syscall.GetsockoptInt(fd, syscall.IPPROTO_TCP, syscall.TCP_KEEPALIVE_THRESHOLD)
	if err != nil {
		t.Fatal(err)
	}
	if time.Duration(tcpKeepAliveThreshold)*time.Millisecond != cfg.Idle {
		t.Fatalf("TCP_KEEPALIVE_THRESHOLD: got %dms; want %v", tcpKeepAliveThreshold, cfg.Idle)
	}

	tcpKeepAliveInterval, err := syscall.GetsockoptInt(fd, syscall.IPPROTO_TCP, syscall_TCP_KEEPINTVL)
	if err != nil {
		t.Fatal(err)
	}
	if time.Duration(tcpKeepAliveInterval)*time.Second != cfg.Interval {
		t.Fatalf("TCP_KEEPINTVL: got %ds; want %v", tcpKeepAliveInterval, cfg.Interval)
	}

	tcpKeepAliveCount, err := syscall.GetsockoptInt(fd, syscall.IPPROTO_TCP, syscall_TCP_KEEPCNT)
	if err != nil {
		t.Fatal(err)
	}
	if tcpKeepAliveCount != cfg.Count {
		t.Fatalf("TCP_KEEPCNT: got %d; want %d", tcpKeepAliveCount, cfg.Count)
	}

	tcpKeepAliveAbortInterval, err := syscall.GetsockoptInt(fd, syscall.IPPROTO_TCP, syscall.TCP_KEEPALIVE_ABORT_THRESHOLD)
	if err != nil {
		t.Fatal(err)
	}
	if time.Duration(tcpKeepAliveAbortInterval)*time.Millisecond != tcpKeepAliveAbortThreshold {
		t.Fatalf("TCP_KEEPALIVE_ABORT_THRESHOLD: got %dms; want %v", tcpKeepAliveAbortInterval, tcpKeepAliveAbortThreshold)
	}
}

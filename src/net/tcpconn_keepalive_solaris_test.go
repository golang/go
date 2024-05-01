// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build solaris && !illumos

package net

import (
	"internal/syscall/unix"
	"syscall"
	"testing"
	"time"
)

func getCurrentKeepAliveSettings(fd fdType) (cfg KeepAliveConfig, err error) {
	tcpKeepAlive, err := syscall.GetsockoptInt(fd, syscall.SOL_SOCKET, syscall.SO_KEEPALIVE)
	if err != nil {
		return
	}

	var (
		tcpKeepAliveIdle         int
		tcpKeepAliveInterval     int
		tcpKeepAliveIdleTime     time.Duration
		tcpKeepAliveIntervalTime time.Duration
		tcpKeepAliveCount        int
	)
	if unix.SupportTCPKeepAliveIdleIntvlCNT() {
		tcpKeepAliveIdle, err = syscall.GetsockoptInt(fd, syscall.IPPROTO_TCP, syscall_TCP_KEEPIDLE)
		if err != nil {
			return
		}
		tcpKeepAliveIdleTime = time.Duration(tcpKeepAliveIdle) * time.Second

		tcpKeepAliveInterval, err = syscall.GetsockoptInt(fd, syscall.IPPROTO_TCP, syscall_TCP_KEEPINTVL)
		if err != nil {
			return
		}
		tcpKeepAliveIntervalTime = time.Duration(tcpKeepAliveInterval) * time.Second

		tcpKeepAliveCount, err = syscall.GetsockoptInt(fd, syscall.IPPROTO_TCP, syscall_TCP_KEEPCNT)
		if err != nil {
			return
		}
	} else {
		tcpKeepAliveIdle, err = syscall.GetsockoptInt(fd, syscall.IPPROTO_TCP, syscall.TCP_KEEPALIVE_THRESHOLD)
		if err != nil {
			return
		}
		tcpKeepAliveIdleTime = time.Duration(tcpKeepAliveIdle) * time.Millisecond

		// TCP_KEEPINTVL and TCP_KEEPCNT are not available on Solaris prior to 11.4,
		// so we have to use the value of TCP_KEEPALIVE_ABORT_THRESHOLD for Interval
		// and 1 for Count to keep this test going.
		tcpKeepAliveInterval, err = syscall.GetsockoptInt(fd, syscall.IPPROTO_TCP, syscall.TCP_KEEPALIVE_ABORT_THRESHOLD)
		if err != nil {
			return
		}
		tcpKeepAliveIntervalTime = time.Duration(tcpKeepAliveInterval) * time.Millisecond
		tcpKeepAliveCount = 1
	}
	cfg = KeepAliveConfig{
		Enable:   tcpKeepAlive != 0,
		Idle:     tcpKeepAliveIdleTime,
		Interval: tcpKeepAliveIntervalTime,
		Count:    tcpKeepAliveCount,
	}
	return
}

func verifyKeepAliveSettings(t *testing.T, fd fdType, oldCfg, cfg KeepAliveConfig) {
	const defaultTcpKeepAliveAbortThreshold = 8 * time.Minute // default value on Solaris

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

	tcpKeepAliveAbortThreshold := defaultTcpKeepAliveAbortThreshold
	if unix.SupportTCPKeepAliveIdleIntvlCNT() {
		// Check out the comment on KeepAliveConfig to understand the following logic.
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
	} else {
		cfg.Interval = cfg.Interval * time.Duration(cfg.Count)
		// Either Interval or Count is set to a negative value, TCP_KEEPALIVE_ABORT_THRESHOLD
		// will remain the default value, so use the old Interval for the subsequent test.
		if cfg.Interval == -1 || cfg.Count == -1 {
			cfg.Interval = oldCfg.Interval
		}
		cfg.Count = 1
		tcpKeepAliveAbortThreshold = cfg.Interval
	}

	tcpKeepAlive, err := syscall.GetsockoptInt(fd, syscall.SOL_SOCKET, syscall.SO_KEEPALIVE)
	if err != nil {
		t.Fatal(err)
	}
	if (tcpKeepAlive != 0) != cfg.Enable {
		t.Fatalf("SO_KEEPALIVE: got %t; want %t", tcpKeepAlive != 0, cfg.Enable)
	}

	// TCP_KEEPALIVE_THRESHOLD and TCP_KEEPALIVE_ABORT_THRESHOLD are both available on Solaris 11.4
	// and previous versions, so we can verify these two options regardless of the kernel version.
	tcpKeepAliveThreshold, err := syscall.GetsockoptInt(fd, syscall.IPPROTO_TCP, syscall.TCP_KEEPALIVE_THRESHOLD)
	if err != nil {
		t.Fatal(err)
	}
	if time.Duration(tcpKeepAliveThreshold)*time.Millisecond != cfg.Idle {
		t.Fatalf("TCP_KEEPIDLE: got %dms; want %v", tcpKeepAliveThreshold, cfg.Idle)
	}

	tcpKeepAliveAbortInterval, err := syscall.GetsockoptInt(fd, syscall.IPPROTO_TCP, syscall.TCP_KEEPALIVE_ABORT_THRESHOLD)
	if err != nil {
		t.Fatal(err)
	}
	if time.Duration(tcpKeepAliveAbortInterval)*time.Millisecond != tcpKeepAliveAbortThreshold {
		t.Fatalf("TCP_KEEPALIVE_ABORT_THRESHOLD: got %dms; want %v", tcpKeepAliveAbortInterval, tcpKeepAliveAbortThreshold)
	}

	if unix.SupportTCPKeepAliveIdleIntvlCNT() {
		tcpKeepAliveIdle, err := syscall.GetsockoptInt(fd, syscall.IPPROTO_TCP, syscall_TCP_KEEPIDLE)
		if err != nil {
			t.Fatal(err)
		}
		if time.Duration(tcpKeepAliveIdle)*time.Second != cfg.Idle {
			t.Fatalf("TCP_KEEPIDLE: got %ds; want %v", tcpKeepAliveIdle, cfg.Idle)
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
	} else {
		if cfg.Count != 1 {
			t.Fatalf("TCP_KEEPCNT: got %d; want 1", cfg.Count)
		}
	}
}

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build aix darwin,!ios dragonfly freebsd linux,!android netbsd openbsd solaris

package time_test

import (
	"os"
	"testing"
	"time"
)

func TestEnvTZUsage(t *testing.T) {
	const env = "TZ"
	tz, ok := os.LookupEnv(env)
	if !ok {
		defer os.Unsetenv(env)
	} else {
		defer os.Setenv(env, tz)
	}
	defer time.ForceUSPacificForTesting()

	localZoneName := "Local"
	// The file may not exist.
	if _, err := os.Stat("/etc/localtime"); os.IsNotExist(err) {
		localZoneName = "UTC"
	}

	cases := []struct {
		nilFlag bool
		tz      string
		local   string
	}{
		// no $TZ means use the system default /etc/localtime.
		{true, "", localZoneName},
		// $TZ="" means use UTC.
		{false, "", "UTC"},
		{false, ":", "UTC"},
		{false, "Asia/Shanghai", "Asia/Shanghai"},
		{false, ":Asia/Shanghai", "Asia/Shanghai"},
		{false, "/etc/localtime", localZoneName},
		{false, ":/etc/localtime", localZoneName},
	}

	for _, c := range cases {
		time.ResetLocalOnceForTest()
		if c.nilFlag {
			os.Unsetenv(env)
		} else {
			os.Setenv(env, c.tz)
		}
		if time.Local.String() != c.local {
			t.Errorf("invalid Local location name for %q: got %q want %q", c.tz, time.Local, c.local)
		}
	}

	time.ResetLocalOnceForTest()
	// The file may not exist on Solaris 2 and IRIX 6.
	path := "/usr/share/zoneinfo/Asia/Shanghai"
	os.Setenv(env, path)
	if _, err := os.Stat(path); os.IsNotExist(err) {
		if time.Local.String() != "UTC" {
			t.Errorf(`invalid path should fallback to UTC: got %q want "UTC"`, time.Local)
		}
		return
	}
	if time.Local.String() != path {
		t.Errorf(`custom path should lead to path itself: got %q want %q`, time.Local, path)
	}

	timeInUTC := time.Date(2009, 1, 1, 12, 0, 0, 0, time.UTC)
	sameTimeInShanghai := time.Date(2009, 1, 1, 20, 0, 0, 0, time.Local)
	if !timeInUTC.Equal(sameTimeInShanghai) {
		t.Errorf("invalid timezone: got %q want %q", timeInUTC, sameTimeInShanghai)
	}

	time.ResetLocalOnceForTest()
	os.Setenv(env, ":"+path)
	if time.Local.String() != path {
		t.Errorf(`custom path should lead to path itself: got %q want %q`, time.Local, path)
	}

	time.ResetLocalOnceForTest()
	os.Setenv(env, path[:len(path)-1])
	if time.Local.String() != "UTC" {
		t.Errorf(`invalid path should fallback to UTC: got %q want "UTC"`, time.Local)
	}
}

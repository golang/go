// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package time_test

import (
	"testing"
	"os"
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

	time.ResetLocalOnceForTest()
	os.Setenv(env, "Asia/Shanghai")
	if time.Local.String() != "Asia/Shanghai" {
		t.Errorf(`invalid Local location name: got %q want "Asia/Shanghai"`, time.Local)
	}

	time.ResetLocalOnceForTest()
	os.Setenv(env, ":Asia/Shanghai")
	if time.Local.String() != "Asia/Shanghai" {
		t.Errorf(`invalid Local location name: got %q want "Asia/Shanghai"`, time.Local)
	}
}

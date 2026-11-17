// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testlogos

import (
	"fmt"
	"internal/asan"
	"internal/goexperiment"
	"internal/msan"
	"internal/race"
	"os"
	"strings"
	"testing"
)

// Inspect the rebuilt executable's compiled constants, not its environment.
func TestSelfBuildConfiguration(t *testing.T) {
	flags := fmt.Sprintf("self-build flags: fast=%t opt=%t check2=%t race=%t asan=%t msan=%t",
		goexperiment.LogosFastCgo, goexperiment.LogosCgoOpt, goexperiment.CgoCheck2,
		race.Enabled, asan.Enabled, msan.Enabled)
	if os.Getenv("LOGOS_SELF_BUILD_CHILD") == "1" {
		fmt.Println(flags)
		return
	}
	binary := selfTestBin(t)
	t.Setenv("LOGOS_SELF_BUILD_CHILD", "1")
	out, err := runCmd(t, "", binary, "-test.run=^TestSelfBuildConfiguration$", "-test.v")
	if err != nil {
		t.Fatalf("rebuilt configuration child: %v\n%s", err, out)
	}
	if !strings.Contains(out, flags+"\n") {
		t.Fatalf("rebuilt configuration differs; want %q:\n%s", flags, out)
	}
}

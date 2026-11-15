// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package buildcfg

import (
	"reflect"
	"strings"
	"testing"
)

func TestLogosExperimentsDefaultOff(t *testing.T) {
	flags, err := ParseGOEXPERIMENT(GOOS, GOARCH, "")
	if err != nil {
		t.Fatalf("ParseGOEXPERIMENT(%q, %q, %q) failed: %v", GOOS, GOARCH, "", err)
	}
	v := reflect.ValueOf(flags.Flags)
	for i := range v.NumField() {
		name := v.Type().Field(i).Name
		if name != "SIMD" && name != "PreemptibleLoops" && !strings.HasPrefix(name, "Logos") {
			continue
		}
		if v.Field(i).Bool() {
			t.Errorf("GOEXPERIMENT %s is on with no flag, want off by default", strings.ToLower(name))
		}
	}
}

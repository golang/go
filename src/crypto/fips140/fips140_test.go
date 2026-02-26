// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fips140

import (
	"internal/godebug"
	"internal/testenv"
	"testing"
)

func TestImmutableGODEBUG(t *testing.T) {
	fips140Enabled := Enabled()
	fips140Setting := godebug.New("fips140")
	fips140SettingValue := fips140Setting.Value()

	tests := []string{
		"fips140=off",
		"fips140=on",
		"fips140=",
		"",
	}
	for _, godebugValue := range tests {
		t.Run(godebugValue, func(t *testing.T) {
			testenv.SetGODEBUG(t, godebugValue)
			if Enabled() != fips140Enabled {
				t.Errorf("Enabled() changed after setting GODEBUG=%s", godebugValue)
			}
			if fips140Setting.Value() != fips140SettingValue {
				t.Errorf("fips140Setting.Value() changed after setting GODEBUG=%s", godebugValue)
			}
		})
	}
}

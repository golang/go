// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fips140

import (
	"internal/godebug"
	"os"
	"testing"
)

func TestImmutableGODEBUG(t *testing.T) {
	defer func(v string) { os.Setenv("GODEBUG", v) }(os.Getenv("GODEBUG"))

	fips140Enabled := Enabled()
	fips140Setting := godebug.New("fips140")
	fips140SettingValue := fips140Setting.Value()

	os.Setenv("GODEBUG", "fips140=off")
	if Enabled() != fips140Enabled {
		t.Errorf("Enabled() changed after setting GODEBUG=fips140=off")
	}
	if fips140Setting.Value() != fips140SettingValue {
		t.Errorf("fips140Setting.Value() changed after setting GODEBUG=fips140=off")
	}

	os.Setenv("GODEBUG", "fips140=on")
	if Enabled() != fips140Enabled {
		t.Errorf("Enabled() changed after setting GODEBUG=fips140=on")
	}
	if fips140Setting.Value() != fips140SettingValue {
		t.Errorf("fips140Setting.Value() changed after setting GODEBUG=fips140=on")
	}

	os.Setenv("GODEBUG", "fips140=")
	if Enabled() != fips140Enabled {
		t.Errorf("Enabled() changed after setting GODEBUG=fips140=")
	}
	if fips140Setting.Value() != fips140SettingValue {
		t.Errorf("fips140Setting.Value() changed after setting GODEBUG=fips140=")
	}

	os.Setenv("GODEBUG", "")
	if Enabled() != fips140Enabled {
		t.Errorf("Enabled() changed after setting GODEBUG=")
	}
	if fips140Setting.Value() != fips140SettingValue {
		t.Errorf("fips140Setting.Value() changed after setting GODEBUG=")
	}
}

// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package time_test

import (
	"internal/syscall/windows/registry"
	"testing"
	. "time"
)

func testZoneAbbr(t *testing.T) {
	t1 := Now()
	// discard nsec
	t1 = Date(t1.Year(), t1.Month(), t1.Day(), t1.Hour(), t1.Minute(), t1.Second(), 0, t1.Location())

	t2, err := Parse(RFC1123, t1.Format(RFC1123))
	if err != nil {
		t.Fatalf("Parse failed: %v", err)
	}
	if t1 != t2 {
		t.Fatalf("t1 (%v) is not equal to t2 (%v)", t1, t2)
	}
}

func TestUSPacificZoneAbbr(t *testing.T) {
	ForceUSPacificFromTZIForTesting() // reset the Once to trigger the race
	defer ForceUSPacificForTesting()
	testZoneAbbr(t)
}

func TestAusZoneAbbr(t *testing.T) {
	ForceAusFromTZIForTesting()
	defer ForceUSPacificForTesting()
	testZoneAbbr(t)
}

func TestToEnglishName(t *testing.T) {
	const want = "Central Europe Standard Time"
	k, err := registry.OpenKey(registry.LOCAL_MACHINE, `SOFTWARE\Microsoft\Windows NT\CurrentVersion\Time Zones\`+want, registry.READ)
	if err != nil {
		t.Fatalf("cannot open CEST time zone information from registry: %s", err)
	}
	defer k.Close()

	var std, dlt string
	if err = registry.LoadRegLoadMUIString(); err == nil {
		// Try MUI_Std and MUI_Dlt first, fallback to Std and Dlt if *any* error occurs
		std, err = k.GetMUIStringValue("MUI_Std")
		if err == nil {
			dlt, err = k.GetMUIStringValue("MUI_Dlt")
		}
	}
	if err != nil { // Fallback to Std and Dlt
		if std, _, err = k.GetStringValue("Std"); err != nil {
			t.Fatalf("cannot read CEST Std registry key: %s", err)
		}
		if dlt, _, err = k.GetStringValue("Dlt"); err != nil {
			t.Fatalf("cannot read CEST Dlt registry key: %s", err)
		}
	}

	name, err := ToEnglishName(std, dlt)
	if err != nil {
		t.Fatalf("toEnglishName failed: %s", err)
	}
	if name != want {
		t.Fatalf("english name: %q, want: %q", name, want)
	}
}

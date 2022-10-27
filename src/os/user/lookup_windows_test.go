// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package user

import (
	"testing"
)

func TestLookupLocalSystem(t *testing.T) {
	// The string representation of the SID for `NT AUTHORITY\SYSTEM`
	const localSystemSID = "S-1-5-18"
	if _, err := LookupId(localSystemSID); err != nil {
		t.Fatalf("LookupId(%q): %v", localSystemSID, err)
	}
}

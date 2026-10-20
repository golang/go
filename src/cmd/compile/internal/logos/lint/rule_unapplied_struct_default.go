// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

// unappliedStructDefault is suspicious::unapplied_struct_default (SPEC Sec
// 1.2, T-104.5): var x T or new(T) for a struct type with field defaults,
// which leaves every default unapplied. The types2 checker finds it while it
// checks the package (types2/structdefault_logos.go, M6-S3), before Run is
// called, and the compiler hands the finding to Run in Input.Reported, which
// applies the severity of the suspicious group and //logos:allow like any
// other finding. The entry has no Check so the finding is reported once, and
// it carries the group and scope. The rule never applies to Go packages,
// which have no field defaults.
var unappliedStructDefault = Rule{
	ID:    "suspicious::unapplied_struct_default",
	Group: Suspicious,
}

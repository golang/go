// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package timeformat defines an Analyzer that checks for the use
// of time.Format or time.Parse calls with a bad format.
//
// # Analyzer timeformat
//
// timeformat: check for calls of (time.Time).Format or time.Parse with 2006-02-01
//
// The timeformat checker looks for time formats with the 2006-02-01 (yyyy-dd-mm)
// format. Internationally, "yyyy-dd-mm" does not occur in common calendar date
// standards, and so it is more likely that 2006-01-02 (yyyy-mm-dd) was intended.
package timeformat

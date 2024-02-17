// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package counter implements a simple counter system for collecting
// totally public telemetry data.
//
// There are two kinds of counters, simple counters and stack counters.
// Simple counters are created by New(<counter-name>).
// Stack counters are created by NewStack(<counter-name>, depth).
// Both are incremented by calling Inc().
//
// Counter files are stored in LocalDir(). Their content can be accessed
// by Parse().
//
// Simple counters are very cheap. Stack counters are more
// expensive, as they require parsing the stack.
// (Stack counters are implemented as a set of regular counters whose names
// are the concatenation of the name and the stack trace. There is an upper
// limit on the size of this name, about 4K bytes. If the name is too long
// the stack will be truncated and "truncated" appended.)
//
// When counter files expire they are turned into reports by the upload package.
// The first time any counter file is created for a user, a random
// day of the week is selected on which counter files will expire.
// For the first week, that day is more than 7 days (but not more than
// two weeks) in the future.
// After that the counter files expire weekly on the same day of
// the week.
package counter

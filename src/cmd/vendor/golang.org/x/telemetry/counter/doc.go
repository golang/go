// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package counter implements a simple counter system for collecting
// totally public telemetry data.
//
// There are two kinds of counters, basic counters and stack counters.
// Basic counters are created by [New].
// Stack counters are created by [NewStack].
// Both are incremented by calling Inc().
//
// Basic counters are very cheap. Stack counters are more expensive, as they
// require parsing the stack. (Stack counters are implemented as basic counters
// whose names are the concatenation of the name and the stack trace. There is
// an upper limit on the size of this name, about 4K bytes. If the name is too
// long the stack will be truncated and "truncated" appended.)
//
// When counter files expire they are turned into reports by the upload
// package. The first time any counter file is created for a user, a random day
// of the week is selected on which counter files will expire. For the first
// week, that day is more than 7 days (but not more than two weeks) in the
// future. After that the counter files expire weekly on the same day of the
// week.
//
// # Counter Naming
//
// Counter names passed to [New] and [NewStack] should follow these
// conventions:
//
//   - Names cannot contain whitespace or newlines.
//
//   - Names must be valid unicode, with no unprintable characters.
//
//   - Names may contain at most one ':'. In the counter "foo:bar", we refer to
//     "foo" as the "chart name" and "bar" as the "bucket name".
//
//   - The '/' character should partition counter names into a hierarchy. The
//     root of this hierarchy should identify the logical entity that "owns"
//     the counter. This could be an application, such as "gopls" in the case
//     of "gopls/client:vscode", or a shared library, such as "crash" in the
//     case of the "crash/crash" counter owned by the crashmonitor library. If
//     the entity name itself contains a '/', that's ok: "cmd/go/flag" is fine.
//
//   - Words should be '-' separated, as in "gopls/completion/errors-latency"
//
//   - Histograms should use bucket names identifying upper bounds with '<'.
//     For example given two counters "gopls/completion/latency:<50ms" and
//     "gopls/completion/latency:<100ms", the "<100ms" bucket counts events
//     with latency in the half-open interval [50ms, 100ms).
package counter

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package telemetry provides an opinionated set of packages that cover the main
// concepts of telemetry in an implementation agnostic way.
// As a library author you should look at
//    stats (for aggregatable measurements)
//    trace (for scoped timing spans)
//    log (for for time based events)
// As a binary author you might look at
//    metric (for methods of aggregating stats)
//    exporter (for methods of exporting the telemetry to external tools)
//    debug (for serving internal http pages of some of the telemetry)
package telemetry

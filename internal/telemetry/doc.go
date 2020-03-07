// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package telemetry provides a set of packages that cover the main
// concepts of telemetry in an implementation agnostic way.
// The interface for libraries that want to expose telemetry is the event
// package.
// As a binary author you might look at exporter for methods of exporting the
// telemetry to external tools.
package telemetry

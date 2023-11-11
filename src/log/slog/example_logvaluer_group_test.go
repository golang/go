// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package slog_test

import "log/slog"

type Name struct {
	First, Last string
}

// LogValue implements slog.LogValuer.
// It returns a group containing the fields of
// the Name, so that they appear together in the log output.
func (n Name) LogValue() slog.Value {
	return slog.GroupValue(
		slog.String("first", n.First),
		slog.String("last", n.Last))
}

func ExampleLogValuer_group() {
	n := Name{"Perry", "Platypus"}
	slog.Info("mission accomplished", "agent", n)

	// JSON Output would look in part like:
	// {
	//     ...
	//     "msg": "mission accomplished",
	//     "agent": {
	//         "first": "Perry",
	//         "last": "Platypus"
	//     }
	// }
}

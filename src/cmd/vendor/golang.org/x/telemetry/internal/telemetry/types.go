// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package telemetry

// Common types and directories used by multiple packages.

// An UploadConfig controls what data is uploaded.
type UploadConfig struct {
	GOOS       []string
	GOARCH     []string
	GoVersion  []string
	SampleRate float64
	Programs   []*ProgramConfig
}

type ProgramConfig struct {
	// the counter names may have to be
	// repeated for each program. (e.g., if the counters are in a package
	// that is used in more than one program.)
	Name     string
	Versions []string        // versions present in a counterconfig
	Counters []CounterConfig `json:",omitempty"`
	Stacks   []CounterConfig `json:",omitempty"`
}

type CounterConfig struct {
	Name  string
	Rate  float64 // If X <= Rate, report this counter
	Depth int     `json:",omitempty"` // for stack counters
}

// A Report is the weekly aggregate of counters.
type Report struct {
	Week     string  // End day this report covers (YYYY-MM-DD)
	LastWeek string  // Week field from latest previous report uploaded
	X        float64 // A random probability used to determine which counters are uploaded
	Programs []*ProgramReport
	Config   string // version of UploadConfig used
}

type ProgramReport struct {
	Program   string // Package path of the program.
	Version   string // Program version. Go version if the program is part of the go distribution. Module version, otherwise.
	GoVersion string // Go version used to build the program.
	GOOS      string
	GOARCH    string
	Counters  map[string]int64
	Stacks    map[string]int64
}

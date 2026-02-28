// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !windows && !darwin

package pprof

import (
	"errors"
	"os"
)

// readMapping reads /proc/self/maps and writes mappings to b.pb.
// It saves the address ranges of the mappings in b.mem for use
// when emitting locations.
func (b *profileBuilder) readMapping() {
	data, _ := os.ReadFile("/proc/self/maps")
	parseProcSelfMaps(data, b.addMapping)
	if len(b.mem) == 0 { // pprof expects a map entry, so fake one.
		b.addMappingEntry(0, 0, 0, "", "", true)
		// TODO(hyangah): make addMapping return *memMap or
		// take a memMap struct, and get rid of addMappingEntry
		// that takes a bunch of positional arguments.
	}
}

func readMainModuleMapping() (start, end uint64, exe, buildID string, err error) {
	return 0, 0, "", "", errors.New("not implemented")
}

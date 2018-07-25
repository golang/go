// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package imports

import "cmd/go/internal/cfg"

var tags map[string]bool

func Tags() map[string]bool {
	if tags == nil {
		tags = loadTags()
	}
	return tags
}

func loadTags() map[string]bool {
	tags := map[string]bool{
		cfg.BuildContext.GOOS:     true,
		cfg.BuildContext.GOARCH:   true,
		cfg.BuildContext.Compiler: true,
	}
	if cfg.BuildContext.CgoEnabled {
		tags["cgo"] = true
	}
	for _, tag := range cfg.BuildContext.BuildTags {
		tags[tag] = true
	}
	for _, tag := range cfg.BuildContext.ReleaseTags {
		tags[tag] = true
	}
	return tags
}

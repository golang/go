// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO: Figure out what gopkg.in should do.

package modfetch

import (
	"cmd/go/internal/modfetch/codehost"
	"cmd/go/internal/modfetch/gitrepo"
	"cmd/go/internal/modfile"
	"fmt"
)

func gopkginLookup(path string) (codehost.Repo, error) {
	root, _, _, _, ok := modfile.ParseGopkgIn(path)
	if !ok {
		return nil, fmt.Errorf("invalid gopkg.in/ path: %q", path)
	}
	return gitrepo.Repo("https://"+root, root)
}

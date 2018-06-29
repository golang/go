// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package googlesource

import (
	"fmt"
	"strings"

	"cmd/go/internal/modfetch/codehost"
	"cmd/go/internal/modfetch/gitrepo"
)

func Lookup(path string) (codehost.Repo, error) {
	i := strings.Index(path, "/")
	if i+1 == len(path) || !strings.HasSuffix(path[:i+1], ".googlesource.com/") {
		return nil, fmt.Errorf("not *.googlesource.com/*")
	}
	j := strings.Index(path[i+1:], "/")
	if j >= 0 {
		path = path[:i+1+j]
	}
	return gitrepo.Repo("https://"+path, path)
}

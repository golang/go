// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"fmt"
	"strings"

	"cmd/go/internal/modfetch/codehost"
	"cmd/go/internal/modfetch/gitrepo"
)

// Lookup returns the code repository enclosing the given module path,
// which must begin with github.com/.
func Lookup(path string) (codehost.Repo, error) {
	f := strings.Split(path, "/")
	if len(f) < 3 || f[0] != "github.com" {
		return nil, fmt.Errorf("github repo must be github.com/org/project")
	}
	path = f[0] + "/" + f[1] + "/" + f[2]
	return gitrepo.Repo("https://"+path, path)
}

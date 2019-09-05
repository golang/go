// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
package modpack

import (
	"cmd/go/internal/semver"
	"errors"
	"fmt"
	"os"
	"strings"
)

type PackInfo struct {
	vcs     string
	owner   string
	name    string
	version string
}

func Validate(project string, version string) error {
	split := strings.Split(project, string(os.PathSeparator))
	if len(split) != 3 {
		return errors.New("project path incorrect")
	}

	if !semver.IsValid(version) {
		return errors.New("version does not satisfy semver")
	}
	return nil
}

func GeneratePackageInfo(project string, version string) (PackInfo, error) {
	split := strings.Split(project, string(os.PathSeparator))
	if len(split) != 3 {
		return PackInfo{}, errors.New("project path incorrect")
	}

	pack := PackInfo{
		vcs:     split[0],
		owner:   split[1],
		name:    split[2],
		version: version,
	}
	return pack, nil
}

func (pkg PackInfo) GetTempFolderName() string {
	return pkg.vcs + string(os.PathSeparator) +
		pkg.owner + string(os.PathSeparator) +
		pkg.name + "@" + pkg.version
}

func (pkg PackInfo) GetModCacheFolderName() string {
	return fmt.Sprintf("%s/%s/%s@%s", pkg.vcs, pkg.owner, pkg.name, pkg.version)
}

func (pkg PackInfo) GetZipFileName() string {
	return pkg.version + ".zip"
}

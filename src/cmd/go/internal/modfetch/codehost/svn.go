// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codehost

import (
	"archive/zip"
	"context"
	"encoding/xml"
	"fmt"
	"io"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"time"

	"cmd/go/internal/base"
)

func svnParseStat(rev, out string) (*RevInfo, error) {
	var log struct {
		Logentry struct {
			Revision int64  `xml:"revision,attr"`
			Date     string `xml:"date"`
		} `xml:"logentry"`
	}
	if err := xml.Unmarshal([]byte(out), &log); err != nil {
		return nil, vcsErrorf("unexpected response from svn log --xml: %v\n%s", err, out)
	}

	t, err := time.Parse(time.RFC3339, log.Logentry.Date)
	if err != nil {
		return nil, vcsErrorf("unexpected response from svn log --xml: %v\n%s", err, out)
	}

	info := &RevInfo{
		Name:    strconv.FormatInt(log.Logentry.Revision, 10),
		Short:   fmt.Sprintf("%012d", log.Logentry.Revision),
		Time:    t.UTC(),
		Version: rev,
	}
	return info, nil
}

func svnReadZip(ctx context.Context, dst io.Writer, workDir, rev, subdir, remote string) (err error) {
	// The subversion CLI doesn't provide a command to write the repository
	// directly to an archive, so we need to export it to the local filesystem
	// instead. Unfortunately, the local filesystem might apply arbitrary
	// normalization to the filenames, so we need to obtain those directly.
	//
	// 'svn export' prints the filenames as they are written, but from reading the
	// svn source code (as of revision 1868933), those filenames are encoded using
	// the system locale rather than preserved byte-for-byte from the origin. For
	// our purposes, that won't do, but we don't want to go mucking around with
	// the user's locale settings either — that could impact error messages, and
	// we don't know what locales the user has available or what LC_* variables
	// their platform supports.
	//
	// Instead, we'll do a two-pass export: first we'll run 'svn list' to get the
	// canonical filenames, then we'll 'svn export' and look for those filenames
	// in the local filesystem. (If there is an encoding problem at that point, we
	// would probably reject the resulting module anyway.)

	remotePath := remote
	if subdir != "" {
		remotePath += "/" + subdir
	}

	release, err := base.AcquireNet()
	if err != nil {
		return err
	}
	out, err := Run(ctx, workDir, []string{
		"svn", "list",
		"--non-interactive",
		"--xml",
		"--incremental",
		"--recursive",
		"--revision", rev,
		"--", remotePath,
	})
	release()
	if err != nil {
		return err
	}

	type listEntry struct {
		Kind string `xml:"kind,attr"`
		Name string `xml:"name"`
		Size int64  `xml:"size"`
	}
	var list struct {
		Entries []listEntry `xml:"entry"`
	}
	if err := xml.Unmarshal(out, &list); err != nil {
		return vcsErrorf("unexpected response from svn list --xml: %v\n%s", err, out)
	}

	exportDir := filepath.Join(workDir, "export")
	// Remove any existing contents from a previous (failed) run.
	if err := os.RemoveAll(exportDir); err != nil {
		return err
	}
	defer os.RemoveAll(exportDir) // best-effort

	release, err = base.AcquireNet()
	if err != nil {
		return err
	}
	_, err = Run(ctx, workDir, []string{
		"svn", "export",
		"--non-interactive",
		"--quiet",

		// Suppress any platform- or host-dependent transformations.
		"--native-eol", "LF",
		"--ignore-externals",
		"--ignore-keywords",

		"--revision", rev,
		"--", remotePath,
		exportDir,
	})
	release()
	if err != nil {
		return err
	}

	// Scrape the exported files out of the filesystem and encode them in the zipfile.

	// “All files in the zip file are expected to be
	// nested in a single top-level directory, whose name is not specified.”
	// We'll (arbitrarily) choose the base of the remote path.
	basePath := path.Join(path.Base(remote), subdir)

	zw := zip.NewWriter(dst)
	for _, e := range list.Entries {
		if e.Kind != "file" {
			continue
		}

		zf, err := zw.Create(path.Join(basePath, e.Name))
		if err != nil {
			return err
		}

		f, err := os.Open(filepath.Join(exportDir, e.Name))
		if err != nil {
			if os.IsNotExist(err) {
				return vcsErrorf("file reported by 'svn list', but not written by 'svn export': %s", e.Name)
			}
			return fmt.Errorf("error opening file created by 'svn export': %v", err)
		}

		n, err := io.Copy(zf, f)
		f.Close()
		if err != nil {
			return err
		}
		if n != e.Size {
			return vcsErrorf("file size differs between 'svn list' and 'svn export': file %s listed as %v bytes, but exported as %v bytes", e.Name, e.Size, n)
		}
	}

	return zw.Close()
}

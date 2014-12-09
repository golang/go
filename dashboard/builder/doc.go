// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*

Go Builder is a continuous build client for the Go project.
It integrates with the Go Dashboard AppEngine application.

Go Builder is intended to run continuously as a background process.

It periodically pulls updates from the Go Mercurial repository.

When a newer revision is found, Go Builder creates a clone of the repository,
runs all.bash, and reports build success or failure to the Go Dashboard.

For a release revision (a change description that matches "release.YYYY-MM-DD"),
Go Builder will create a tar.gz archive of the GOROOT and deliver it to the
Go Google Code project's downloads section.

Usage:

  gobuilder goos-goarch...

  Several goos-goarch combinations can be provided, and the builder will
  build them in serial.

Optional flags:

  -dashboard="godashboard.appspot.com": Go Dashboard Host
    The location of the Go Dashboard application to which Go Builder will
    report its results.

  -release: Build and deliver binary release archive

  -rev=N: Build revision N and exit

  -cmd="./all.bash": Build command (specify absolute or relative to go/src)

  -v: Verbose logging

  -external: External package builder mode (will not report Go build
     state to dashboard or issue releases)

The key file should be located at $HOME/.gobuildkey or, for a builder-specific
key, $HOME/.gobuildkey-$BUILDER (eg, $HOME/.gobuildkey-linux-amd64).

The build key file is a text file of the format:

  godashboard-key
  googlecode-username
  googlecode-password

If the Google Code credentials are not provided the archival step
will be skipped.

*/
package main // import "golang.org/x/tools/dashboard/builder"

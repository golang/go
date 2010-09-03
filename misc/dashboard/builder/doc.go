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

For a successful build, Go Builder will also run benchmarks 
(cd $GOROOT/src/pkg; make bench) and send the results to the Go Dashboard.

For release revision (a change description that matches "release.YYYY-MM-DD"),
Go Builder will create a tar.gz archive of the GOROOT and deliver it to the
Go Google Code project's downloads section.

Command-line options (and defaults):

  -goarch="": $GOARCH
  -goos="": $GOOS
    The target architecture and operating system of this build client.

  -goroot="": $GOROOT
    A persistent Go checkout. Go Builder will periodically run 'hg pull -u' 
    from this location and use it as a source repository when cloning a
    revision to be built.

  -path="": Build Path
    The base path in which building, testing, and archival will occur,
    such as "/tmp/build".  This can be considered volatile.

  -keyfile="": Key File
    The file containing the build key and Google Code credentials.
    It is a text file of the format:

      godashboard-key
      googlecode-username
      googlecode-password

    If the Google Code credentials are not provided the archival step
    will be skipped.

  -host="godashboard.appspot.com": Go Dashboard Host
    The location of the Go Dashboard application to which Go Builder will
    report its results.

  -pybin="/usr/bin/python": Python Binary
  -hgbin="/usr/local/bin/hg": Mercurial Binary
    These name the local Python and Mercurial binaries.
    (Python is required only to run the Google Code uploader script, found
     at $GOROOT/misc/dashboard/googlecode_upload.py.)

*/
package documentation

Want to contribute? Great: read the page (including the small print at the end).

# Before you contribute

As an individual, sign the [Google Individual Contributor License
Agreement](https://cla.developers.google.com/about/google-individual) (CLA)
online. This is required for any of your code to be accepted.

Before you start working on a larger contribution, get in touch with us first
through the issue tracker with your idea so that we can help out and possibly
guide you. Coordinating up front makes it much easier to avoid frustration later
on.

# Development

Make sure `GOPATH` is set in your current shell. The common way is to have
something like `export GOPATH=$HOME/gocode` in your `.bashrc` file so that it's
automatically set in all console sessions.

To get the source code, run

```
go get github.com/google/pprof
```

To run the tests, do

```
cd $GOPATH/src/github.com/google/pprof
go test -v ./...
```

When you wish to work with your own fork of the source (which is required to be
able to create a pull request), you'll want to get your fork repo as another Git
remote in the same `github.com/google/pprof` directory. Otherwise, if you'll `go
get` your fork directly, you'll be getting errors like `use of internal package
not allowed` when running tests.  To set up the remote do something like

```
cd $GOPATH/src/github.com/google/pprof
git remote add aalexand git@github.com:aalexand/pprof.git
git fetch aalexand
git checkout -b my-new-feature
# hack hack hack
go test -v ./...
git commit -a -m "Add new feature."
git push aalexand
```

where `aalexand` is your GitHub user ID. Then proceed to the GitHub UI to send a
code review.

# Code reviews

All submissions, including submissions by project members, require review.
We use GitHub pull requests for this purpose.

The pprof source code is in Go with a bit of JavaScript, CSS and HTML. If you
are new to Go, read [Effective Go](https://golang.org/doc/effective_go.html) and
the [summary on typical comments during Go code
reviews](https://github.com/golang/go/wiki/CodeReviewComments).

Cover all new functionality with tests. Enable Travis on your forked repo,
enable builds of branches and make sure Travis is happily green for the branch
with your changes.

The code coverage is measured for each pull request. The code coverage is
expected to go up with every change.

Pull requests not meeting the above guidelines will get less attention than good
ones, so make sure your submissions are high quality.

# The small print

Contributions made by corporations are covered by a different agreement than the
one above, the [Software Grant and Corporate Contributor License
Agreement](https://cla.developers.google.com/about/google-corporate).

# README.md

This directory holds build scripts for unofficial, unsupported
distributions of Go+BoringCrypto.

## Version strings

The distribution name for a Go+BoringCrypto release has the form `<GoVersion>b<BoringCryptoVersion>`,
where `<GoVersion>` is the Go version the release is based on, and `<BoringCryptoVersion>` is
an integer that increments each time there is a new release with different BoringCrypto bits.
The `<BoringCryptoVersion>` is stored in the `VERSION` file in this directory.

For example, the first release is based on Go 1.8.3 is `go1.8.3b1`.
If the BoringCrypto bits are updated, the next would be `go1.8.3b2`.
If, after that, Go 1.9 is released and the same BoringCrypto code added to it,
that would result in `go1.9b2`. There would likely not be a `go1.9b1`,
since that would indicate Go 1.9 with the older BoringCrypto code.

## Releases

The `build.release` script prepares a binary release and publishes it in Google Cloud Storage
at `gs://go-boringcrypto/`, making it available for download at
`https://go-boringcrypto.storage.googleapis.com/<FILE>`.
The script records each published release in the `RELEASES` file in this directory.

The `build.docker` script, which must be run after `build.release`, prepares a Docker image
and publishes it on hub.docker.com in the goboring organization.
`go1.8.3b1` is published as `goboring/golang:1.8.3b1`.

## Release process

Development is done on the dev.boringcrypto branch, which tracks
master. Releases are cut from dev.boringcrypto.go1.X branches,
which are BoringCrypto backported to the Go 1.X release branches.
To issue new BoringCrypto releases based on Go 1.X:

1. If the BoringCrypto bits have been updated, increment the
   number in `VERSION`, send that change out as a CL for review,
   get it committed to dev.boringcrypto, and run `git sync`.

2. Change to the dev.boringcrypto.go1.X branch and cherry-pick
   all BoringCrypto updates, including the update of the
   `VERSION` file. If desired, merge release-branch.go1.X into
   dev.boringcrypto.go1.X. Mail them out and get them committed.

3. **Back on the dev.boringcrypto branch**, run `git fetch`,
   `make.bash` and then `build.release dev.boringcrypto.go1.X`.
   The script will determine the base Go version and the
   BoringCrypto version, build a release, and upload it.

4. Run `build.docker`, which will build and upload a Docker image
   from the latest release.

5. Send out a CL with the updated `RELEASES` file and get it
   committed to dev.boringcrypto.

## Building from Docker

A Dockerfile that starts with `FROM golang:1.8.3` can switch
to `FROM goboring/golang:1.8.3b2` (see [goboring/golang on Docker Hub](https://hub.docker.com/r/goboring/golang/))
and should need no other modifications.

## Building from Bazel

Starting from [bazelbuild/rules_go](https://github.com/bazelbuild/rules_go)
tag 0.7.1, simply download the BoringCrypto-enabled Go SDK using
`go_download_sdk()` before calling `go_register_toolchains()`.

For example, to use Go 1.9.3 with BoringCrypto on Linux, use the following lines
in `WORKSPACE`:
```python
load("@io_bazel_rules_go//go:def.bzl", "go_rules_dependencies", "go_download_sdk", "go_register_toolchains")

go_rules_dependencies()

go_download_sdk(
    name = "go_sdk",
    sdks = {
       "linux_amd64": ("go1.9.3b4.linux-amd64.tar.gz", "db1997b2454a2f27669b849d2d2cafb247a55128d53da678f06cb409310d6660"),
    },
    urls = ["https://storage.googleapis.com/go-boringcrypto/{}"],
)

go_register_toolchains()
```

**Note**: you must *not* enable `pure` mode, since cgo must be enabled. To
ensure that binaries are linked with BoringCrypto, you can set `pure = "off"` on
all relevant `go_binary` rules.

## Caveat

BoringCrypto is used for a given build only in limited circumstances:

  - The build must be GOOS=linux, GOARCH=amd64.
  - The build must have cgo enabled.
  - The android build tag must not be specified.
  - The cmd_go_bootstrap build tag must not be specified.

The version string reported by `runtime.Version` does not indicate that BoringCrypto
was actually used for the build. For example, linux/386 and non-cgo linux/amd64 binaries
will report a version of `go1.8.3b2` but not be using BoringCrypto.

To check whether a given binary is using BoringCrypto, run `go tool nm` on it and check
that it has symbols named `*_Cfunc__goboringcrypto_*`.

The program [rsc.io/goversion](https://godoc.org/rsc.io/goversion) will report the
crypto implementation used by a given binary when invoked with the `-crypto` flag.

## Ports {#ports}

### Darwin

<!-- go.dev/issue/75836 -->

Go 1.26 is the last release that will run on macOS 12 Monterey. Go 1.27 will require macOS 13 Ventura or later.

### FreeBSD

<!-- go.dev/issue/76475 -->

The freebsd/riscv64 port (`GOOS=freebsd GOARCH=riscv64`) has been marked broken.
See [issue 76475](/issue/76475) for details.

### Windows

<!-- go.dev/issue/71671 -->

As [announced](/doc/go1.25#windows) in the Go 1.25 release notes, the [broken](/doc/go1.24#windows) 32-bit windows/arm port (`GOOS=windows` `GOARCH=arm`) is removed.

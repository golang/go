---
title: "Go, Backwards Compatibility, and GODEBUG"
layout: article
---

<!--
This document is kept in the Go repo, not x/website,
because it documents the full list of known GODEBUG settings,
which are tied to a specific release.
-->

## Introduction {#intro}

Go's emphasis on backwards compatibility is one of its key strengths.
There are, however, times when we cannot maintain complete compatibility.
If code depends on buggy (including insecure) behavior,
then fixing the bug will break that code.
New features can also have similar impacts:
enabling the HTTP/2 use by the HTTP client broke programs
connecting to servers with buggy HTTP/2 implementations.
These kinds of changes are unavoidable and
[permitted by the Go 1 compatibility rules](/doc/go1compat).
Even so, Go provides a mechanism called GODEBUG to
reduce the impact such changes have on Go developers
using newer toolchains to compile old code.

A GODEBUG setting is a `key=value` pair
that controls the execution of certain parts of a Go program.
The environment variable `GODEBUG`
can hold a comma-separated list of these settings.
For example, if a Go program is running in an environment that contains

	GODEBUG=http2client=0,http2server=0

then that Go program will disable the use of HTTP/2 by default in both
the HTTP client and the HTTP server.
It is also possible to set the default `GODEBUG` for a given program
(discussed below).

When preparing any change that is permitted by Go 1 compatibility
but may nonetheless break some existing programs,
we first engineer the change to keep as many existing programs working as possible.
For the remaining programs,
we define a new GODEBUG setting that
allows individual programs to opt back in to the old behavior.
A GODEBUG setting may not be added if doing so is infeasible,
but that should be extremely rare.

GODEBUG settings added for compatibility will be maintained
for a minimum of two years (four Go releases).
Some, such as `http2client` and `http2server`,
will be maintained much longer, even indefinitely.

When possible, each GODEBUG setting has an associated
[runtime/metrics](/pkg/runtime/metrics/) counter
named `/godebug/non-default-behavior/<name>:events`
that counts the number of times a particular program's
behavior has changed based on a non-default value
for that setting.
For example, when `GODEBUG=http2client=0` is set,
`/godebug/non-default-behavior/http2client:events`
counts the number of HTTP transports that the program
has configured without HTTP/2 support.

## Default GODEBUG Values {#default}

When a GODEBUG setting is not listed in the environment variable,
its value is derived from three sources:
the defaults for the Go toolchain used to build the program,
amended to match the Go version listed in `go.mod`,
and then overridden by explicit `//go:debug` lines in the program.

The [GODEBUG History](#history) gives the exact defaults for each Go toolchain version.
For example, Go 1.21 introduces the `panicnil` setting,
controlling whether `panic(nil)` is allowed;
it defaults to `panicnil=0`, making `panic(nil)` a run-time error.
Using `panicnil=1` restores the behavior of Go 1.20 and earlier.

When compiling a work module or workspace that declares
an older Go version, the Go toolchain amends its defaults
to match that older Go version as closely as possible.
For example, when a Go 1.21 toolchain compiles a program,
if the work module's `go.mod` or the workspace's `go.work`
says `go` `1.20`, then the program defaults to `panicnil=1`,
matching Go 1.20 instead of Go 1.21.

Because this method of setting GODEBUG defaults was introduced only in Go 1.21,
programs listing versions of Go earlier than Go 1.20 are configured to match Go 1.20,
not the older version.

To override these defaults, a main package's source files
can include one or more `//go:debug` directives at the top of the file
(preceding the `package` statement).
Continuing the `panicnil` example, if the module or workspace is updated
to say `go` `1.21`, the program can opt back into the old `panic(nil)`
behavior by including this directive:

	//go:debug panicnil=1

Starting in Go 1.21, the Go toolchain treats a `//go:debug` directive
with an unrecognized GODEBUG setting as an invalid program.
Programs with more than one `//go:debug` line for a given setting
are also treated as invalid.
(Older toolchains ignore `//go:debug` directives entirely.)

The defaults that will be compiled into a main package
are reported by the command:

{{raw `
	go list -f '{{.DefaultGODEBUG}}' my/main/package
`}}

Only differences from the base Go toolchain defaults are reported.

When testing a package, `//go:debug` lines in the `*_test.go`
files are treated as directives for the test's main package.
In any other context, `//go:debug` lines are ignored by the toolchain;
`go` `vet` reports such lines as misplaced.

## GODEBUG History {#history}

This section documents the GODEBUG settings introduced and removed in each major Go release
for compatibility reasons.
Packages or programs may define additional settings for internal debugging purposes;
for example,
see the [runtime documentation](/pkg/runtime#hdr-Environment_Variables)
and the [go command documentation](/cmd/go#hdr-Build_and_test_caching).

### Go 1.23

Go 1.23 introduces the use of caching by http.Dir when the file modification time has not changed. 
Setting `httpdircache` to 0 restores the old behavior.

### Go 1.22

Go 1.22 adds a configurable limit to control the maximum acceptable RSA key size
that can be used in TLS handshakes, controlled by the [`tlsmaxrsasize` setting](/pkg/crypto/tls#Conn.Handshake).
The default is tlsmaxrsasize=8192, limiting RSA to 8192-bit keys. To avoid
denial of service attacks, this setting and default was backported to Go
1.19.13, Go 1.20.8, and Go 1.21.1.

Go 1.22 made it an error for a request or response read by a net/http
client or server to have an empty Content-Length header.
This behavior is controlled by the `httplaxcontentlength` setting.

Go 1.22 changed the behavior of ServeMux to accept extended
patterns and unescape both patterns and request paths by segment.
This behavior can be controlled by the
[`httpmuxgo121` setting](/pkg/net/http/#ServeMux).

Go 1.22 added the [Alias type](/pkg/go/types#Alias) to [go/types](/pkg/go/types)
for the explicit representation of [type aliases](/ref/spec#Type_declarations).
Whether the type checker produces `Alias` types or not is controlled by the
[`gotypesalias` setting](/pkg/go/types#Alias).
For Go 1.22 it defaults to `gotypesalias=0`.
For Go 1.23, `gotypealias=1` will become the default.
This setting will be removed in a future release, Go 1.24 at the earliest.

Go 1.22 changed the default minimum TLS version supported by both servers
and clients to TLS 1.2. The default can be reverted to TLS 1.0 using the
[`tls10server` setting](/pkg/crypto/tls/#Config).

Go 1.22 changed the default TLS cipher suites used by clients and servers when
not explicitly configured, removing the cipher suites which used RSA based key
exchange. The default can be revert using the [`tlsrsakex` setting](/pkg/crypto/tls/#Config).

Go 1.22 disabled
[`ConnectionState.ExportKeyingMaterial`](/pkg/crypto/tls/#ConnectionState.ExportKeyingMaterial)
when the connection supports neither TLS 1.3 nor Extended Master Secret
(implemented in Go 1.21). It can be reenabled with the [`tlsunsafeekm`
setting](/pkg/crypto/tls/#ConnectionState.ExportKeyingMaterial).

Go 1.22 changed how the runtime interacts with transparent huge pages on Linux.
In particular, a common default Linux kernel configuration can result in
significant memory overheads, and Go 1.22 no longer works around this default.
To work around this issue without adjusting kernel settings, transparent huge
pages can be disabled for Go memory with the
[`disablethp` setting](/pkg/runtime#hdr-Environment_Variable).
This behavior was backported to Go 1.21.1, but the setting is only available
starting with Go 1.21.6.
This setting may be removed in a future release, and users impacted by this issue
should adjust their Linux configuration according to the recommendations in the
[GC guide](/doc/gc-guide#Linux_transparent_huge_pages), or switch to a Linux
distribution that disables transparent huge pages altogether.

Go 1.22 added contention on runtime-internal locks to the [`mutex`
profile](/pkg/runtime/pprof#Profile). Contention on these locks is always
reported at `runtime._LostContendedRuntimeLock`. Complete stack traces of
runtime locks can be enabled with the [`runtimecontentionstacks`
setting](/pkg/runtime#hdr-Environment_Variable). These stack traces have
non-standard semantics, see setting documentation for details.

Go 1.22 added a new [`crypto/x509.Certificate`](/pkg/crypto/x509/#Certificate)
field, [`Policies`](/pkg/crypto/x509/#Certificate.Policies), which supports
certificate policy OIDs with components larger than 31 bits. By default this
field is only used during parsing, when it is populated with policy OIDs, but
not used during marshaling. It can be used to marshal these larger OIDs, instead
of the existing PolicyIdentifiers field, by using the
[`x509usepolicies` setting.](/pkg/crypto/x509/#CreateCertificate).


### Go 1.21

Go 1.21 made it a run-time error to call `panic` with a nil interface value,
controlled by the [`panicnil` setting](/pkg/builtin/#panic).

Go 1.21 made it an error for html/template actions to appear inside of an ECMAScript 6
template literal, controlled by the
[`jstmpllitinterp` setting](/pkg/html/template#hdr-Security_Model).
This behavior was backported to Go 1.19.8+ and Go 1.20.3+.

Go 1.21 introduced a limit on the maximum number of MIME headers and multipart
forms, controlled by the
[`multipartmaxheaders` and `multipartmaxparts` settings](/pkg/mime/multipart#hdr-Limits)
respectively.
This behavior was backported to Go 1.19.8+ and Go 1.20.3+.

Go 1.21 adds the support of Multipath TCP but it is only used if the application
explicitly asked for it. This behavior can be controlled by the
[`multipathtcp` setting](/pkg/net#Dialer.SetMultipathTCP).

There is no plan to remove any of these settings.

### Go 1.20

Go 1.20 introduced support for rejecting insecure paths in tar and zip archives,
controlled by the [`tarinsecurepath` setting](/pkg/archive/tar/#Reader.Next)
and the [`zipinsecurepath` setting](/pkg/archive/zip/#NewReader).
These default to `tarinsecurepath=1` and `zipinsecurepath=1`,
preserving the behavior of earlier versions of Go.
A future version of Go may change the defaults to
`tarinsecurepath=0` and `zipinsecurepath=0`.

Go 1.20 introduced automatic seeding of the
[`math/rand`](/pkg/math/rand) global random number generator,
controlled by the [`randautoseed` setting](/pkg/math/rand/#Seed).

Go 1.20 introduced the concept of fallback roots for use during certificate verification,
controlled by the [`x509usefallbackroots` setting](/pkg/crypto/x509/#SetFallbackRoots).

Go 1.20 removed the preinstalled `.a` files for the standard library
from the Go distribution.
Installations now build and cache the standard library like
packages in other modules.
The [`installgoroot` setting](/cmd/go#hdr-Compile_and_install_packages_and_dependencies)
restores the installation and use of preinstalled `.a` files.

There is no plan to remove any of these settings.

### Go 1.19

Go 1.19 made it an error for path lookups to resolve to binaries in the current directory,
controlled by the [`execerrdot` setting](/pkg/os/exec#hdr-Executables_in_the_current_directory).
There is no plan to remove this setting.

### Go 1.18

Go 1.18 removed support for SHA1 in most X.509 certificates,
controlled by the [`x509sha1` setting](/pkg/crypto/x509#InsecureAlgorithmError).
This setting will be removed in a future release, Go 1.22 at the earliest.

### Go 1.10

Go 1.10 changed how build caching worked and added test caching, along
with the [`gocacheverify`, `gocachehash`, and `gocachetest` settings](/cmd/go/#hdr-Build_and_test_caching).
There is no plan to remove these settings.

### Go 1.6

Go 1.6 introduced transparent support for HTTP/2,
controlled by the [`http2client`, `http2server`, and `http2debug` settings](/pkg/net/http/#hdr-HTTP_2).
There is no plan to remove these settings.

### Go 1.5

Go 1.5 introduced a pure Go DNS resolver,
controlled by the [`netdns` setting](/pkg/net/#hdr-Name_Resolution).
There is no plan to remove this setting.

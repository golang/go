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
Unrecognized settings in the `GODEBUG` environment variable are ignored.
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

To override these defaults, starting in Go 1.23, the work module's `go.mod`
or the workspace's `go.work` can list one or more `godebug` lines:

	godebug (
		default=go1.21
		panicnil=1
		asynctimerchan=0
	)

The special key `default` indicates a Go version to take unspecified
settings from. This allows setting the GODEBUG defaults separately
from the Go language version in the module.
In this example, the program is asking for Go 1.21 semantics and
then asking for the old pre-Go 1.21 `panic(nil)` behavior and the
new Go 1.23 `asynctimerchan=0` behavior.

Only the work module's `go.mod` is consulted for `godebug` directives.
Any directives in required dependency modules are ignored.
It is an error to list a `godebug` with an unrecognized setting.
(Toolchains older than Go 1.23 reject all `godebug` lines, since they do not
understand `godebug` at all.) When a workspace is in use, `godebug`
directives in `go.mod` files are ignored, and `go.work` will be consulted
for `godebug` directives instead.

The defaults from the `go` and `godebug` lines apply to all main
packages that are built. For more fine-grained control,
starting in Go 1.21, a main package's source files
can include one or more `//go:debug` directives at the top of the file
(preceding the `package` statement).
The `godebug` lines in the previous example would be written:

	//go:debug default=go1.21
	//go:debug panicnil=1
	//go:debug asynctimerchan=0

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

### Go 1.26

Go 1.26 added a new `httpcookiemaxnum` setting that controls the maximum number
of cookies that net/http will accept when parsing HTTP headers. If the number of
cookie in a header exceeds the number set in `httpcookiemaxnum`, cookie parsing
will fail early. The default value is `httpcookiemaxnum=3000`. Setting
`httpcookiemaxnum=0` will allow the cookie parsing to accept an indefinite
number of cookies. To avoid denial of service attacks, this setting and default
was backported to Go 1.25.2 and Go 1.24.8.

Go 1.26 added a new `urlstrictcolons` setting that controls whether `net/url.Parse`
allows malformed hostnames containing colons outside of a bracketed IPv6 address.
The default `urlstrictcolons=1` rejects URLs such as `http://localhost:1:2` or `http://::1/`.
Colons are permitted as part of a bracketed IPv6 address, such as `http://[::1]/`.

Go 1.26 added a new `tracebacklabels` setting that controls the inclusion of
goroutine labels set through the the `runtime/pprof` package. Setting `tracebacklabels=1`
includes these key/value pairs in the goroutine status header of runtime
tracebacks and debug=2 runtime/pprof stack dumps. This format may change in the future.
(see go.dev/issue/76349)

### Go 1.25

Go 1.25 added a new `decoratemappings` setting that controls whether the Go
runtime annotates OS anonymous memory mappings with context about their
purpose. These annotations appear in /proc/self/maps and /proc/self/smaps as
"[anon: Go: ...]". This setting is only used on Linux. For Go 1.25, it defaults
to `decoratemappings=1`, enabling annotations. Using `decoratemappings=0`
reverts to the pre-Go 1.25 behavior. This setting is fixed at program startup
time, and can't be modified by changing the `GODEBUG` environment variable
after the program starts.

Go 1.25 added a new `embedfollowsymlinks` setting that controls whether the
Go command will follow symlinks to regular files embedding files.
The default value `embedfollowsymlinks=0` does not allow following
symlinks. `embedfollowsymlinks=1` will allow following symlinks.

Go 1.25 added a new `containermaxprocs` setting that controls whether the Go
runtime will consider cgroup CPU limits when setting the default GOMAXPROCS.
The default value `containermaxprocs=1` will use cgroup limits in addition to
the total logical CPU count and CPU affinity. `containermaxprocs=0` will
disable consideration of cgroup limits. This setting only affects Linux.

Go 1.25 added a new `updatemaxprocs` setting that controls whether the Go
runtime will periodically update GOMAXPROCS for new CPU affinity or cgroup
limits. The default value `updatemaxprocs=1` will enable periodic updates.
`updatemaxprocs=0` will disable periodic updates.

Go 1.25 disabled SHA-1 signature algorithms in TLS 1.2 according to RFC 9155.
The default can be reverted using the `tlssha1=1` setting.

Go 1.25 switched to SHA-256 to fill in missing SubjectKeyId in
crypto/x509.CreateCertificate. The setting `x509sha256skid=0` reverts to SHA-1.

Go 1.25 corrected the semantics of contention reports for runtime-internal locks,
and so removed the [`runtimecontentionstacks` setting](/pkg/runtime#hdr-Environment_Variables).

Go 1.25 (starting with Go 1.25 RC 2) disabled build information stamping when
multiple VCS are detected due to concerns around VCS injection attacks. This
behavior and setting was backported to Go 1.24.5 and Go 1.23.11. This behavior
can be renabled with the setting `allowmultiplevcs=1`.

### Go 1.24

Go 1.24 added a new `fips140` setting that controls whether the Go
Cryptographic Module operates in FIPS 140-3 mode.
The possible values are:
- "off": no special support for FIPS 140-3 mode. This is the default.
- "on": the Go Cryptographic Module operates in FIPS 140-3 mode.
- "only": like "on", but cryptographic algorithms not approved by
  FIPS 140-3 return an error or panic.
For more information, see [FIPS 140-3 Compliance](/doc/security/fips140).
This setting is fixed at program startup time, and can't be modified
by changing the `GODEBUG` environment variable after the program starts.

Go 1.24 changed the global [`math/rand.Seed`](/pkg/math/rand/#Seed) to be a
no-op. This behavior is controlled by the `randseednop` setting.
For Go 1.24 it defaults to `randseednop=1`.
Using `randseednop=0` reverts to the pre-Go 1.24 behavior.

Go 1.24 added new values for the `multipathtcp` setting.
The possible values for `multipathtcp` are now:
- "0": disable MPTCP on dialers and listeners by default
- "1": enable MPTCP on dialers and listeners by default
- "2": enable MPTCP on listeners only by default
- "3": enable MPTCP on dialers only by default

For Go 1.24, it now defaults to multipathtcp="2", thus
enabled by default on listeners. Using multipathtcp="0" reverts to the
pre-Go 1.24 behavior.

Go 1.24 changed the behavior of `go test -json` to emit build errors as JSON
instead of text.
These new JSON events are distinguished by new `Action` values,
but can still cause problems with CI systems that aren't robust to these events.
This behavior can be controlled with the `gotestjsonbuildtext` setting.
Using `gotestjsonbuildtext=1` restores the 1.23 behavior.
This setting will be removed in a future release, Go 1.28 at the earliest.

Go 1.24 changed [`crypto/rsa`](/pkg/crypto/rsa) to require RSA keys to be at
least 1024 bits. This behavior can be controlled with the `rsa1024min` setting.
Using `rsa1024min=0` restores the Go 1.23 behavior.

Go 1.24 introduced a mechanism for enabling platform specific Data Independent
Timing (DIT) modes in the [`crypto/subtle`](/pkg/crypto/subtle) package. This
mode can be enabled for an entire program with the `dataindependenttiming` setting.
For Go 1.24 it defaults to `dataindependenttiming=0`. There is no change in default
behavior from Go 1.23 when `dataindependenttiming` is unset.
Using `dataindependenttiming=1` enables the DIT mode for the entire Go program.
When enabled, DIT will be enabled when calling into C from Go. When enabled,
calling into Go code from C will enable DIT, and disable it before returning to
C if it was not enabled when Go code was entered.
This currently only affects arm64 programs. For all other platforms it is a no-op.

Go 1.24 removed the `x509sha1` setting.  `crypto/x509` no longer supports verifying
signatures on certificates that use SHA-1 based signature algorithms.

Go 1.24 changes the default value of the [`x509usepolicies`
setting.](/pkg/crypto/x509/#CreateCertificate) from `0` to `1`. When marshalling
certificates, policies are now taken from the
[`Certificate.Policies`](/pkg/crypto/x509/#Certificate.Policies) field rather
than the
[`Certificate.PolicyIdentifiers`](/pkg/crypto/x509/#Certificate.PolicyIdentifiers)
field by default.

Go 1.24 enabled the post-quantum key exchange mechanism
X25519MLKEM768 by default. The default can be reverted using the
[`tlsmlkem` setting](/pkg/crypto/tls/#Config.CurvePreferences).
This can be useful when dealing with buggy TLS servers that do not handle large records correctly,
causing a timeout during the handshake (see [TLS post-quantum TL;DR fail](https://tldr.fail/)).
Go 1.24 also removed X25519Kyber768Draft00 and the Go 1.23 `tlskyber` setting.

Go 1.24 made [`ParsePKCS1PrivateKey`](/pkg/crypto/x509/#ParsePKCS1PrivateKey)
use and validate the CRT parameters in the encoded private key. This behavior
can be controlled with the `x509rsacrt` setting. Using `x509rsacrt=0` restores
the Go 1.23 behavior.

### Go 1.23

Go 1.23 changed the channels created by package time to be unbuffered
(synchronous), which makes correct use of the [`Timer.Stop`](/pkg/time/#Timer.Stop)
and [`Timer.Reset`](/pkg/time/#Timer.Reset) method results much easier.
The [`asynctimerchan` setting](/pkg/time/#NewTimer) disables this change.
There are no runtime metrics for this change,
This setting may be removed in a future release, Go 1.27 at the earliest.

Go 1.23 changed the mode bits reported by [`os.Lstat`](/pkg/os#Lstat) and [`os.Stat`](/pkg/os#Stat)
for reparse points, which can be controlled with the `winsymlink` setting.
As of Go 1.23 (`winsymlink=1`), mount points no longer have [`os.ModeSymlink`](/pkg/os#ModeSymlink)
set, and reparse points that are not symlinks, Unix sockets, or dedup files now
always have [`os.ModeIrregular`](/pkg/os#ModeIrregular) set. As a result of these changes,
[`filepath.EvalSymlinks`](/pkg/path/filepath#EvalSymlinks) no longer evaluates
mount points, which was a source of many inconsistencies and bugs.
At previous versions (`winsymlink=0`), mount points are treated as symlinks,
and other reparse points with non-default [`os.ModeType`](/pkg/os#ModeType) bits
(such as [`os.ModeDir`](/pkg/os#ModeDir)) do not have the `ModeIrregular` bit set.

Go 1.23 changed [`os.Readlink`](/pkg/os#Readlink) and [`filepath.EvalSymlinks`](/pkg/path/filepath#EvalSymlinks)
to avoid trying to normalize volumes to drive letters, which was not always even possible.
This behavior is controlled by the `winreadlinkvolume` setting.
For Go 1.23, it defaults to `winreadlinkvolume=1`.
Previous versions default to `winreadlinkvolume=0`.

Go 1.23 enabled the experimental post-quantum key exchange mechanism
X25519Kyber768Draft00 by default. The default can be reverted using the
[`tlskyber` setting](/pkg/crypto/tls/#Config.CurvePreferences).
This can be useful when dealing with buggy TLS servers that do not handle large records correctly,
causing a timeout during the handshake (see [TLS post-quantum TL;DR fail](https://tldr.fail/)).

Go 1.23 changed the behavior of
[crypto/x509.ParseCertificate](/pkg/crypto/x509/#ParseCertificate) to reject
serial numbers that are negative. This change can be reverted with
the [`x509negativeserial` setting](/pkg/crypto/x509/#ParseCertificate).

Go 1.23 re-enabled support in html/template for ECMAScript 6 template literals by default.
The [`jstmpllitinterp` setting](/pkg/html/template#hdr-Security_Model) no longer has
any effect.

Go 1.23 changed the default TLS cipher suites used by clients and servers when
not explicitly configured, removing 3DES cipher suites. The default can be reverted
using the [`tls3des` setting](/pkg/crypto/tls/#Config.CipherSuites).

Go 1.23 changed the behavior of [`tls.X509KeyPair`](/pkg/crypto/tls#X509KeyPair)
and [`tls.LoadX509KeyPair`](/pkg/crypto/tls#LoadX509KeyPair) to populate the
Leaf field of the returned [`tls.Certificate`](/pkg/crypto/tls#Certificate).
This behavior is controlled by the `x509keypairleaf` setting. For Go 1.23, it
defaults to `x509keypairleaf=1`. Previous versions default to
`x509keypairleaf=0`.

Go 1.23 changed
[`net/http.ServeContent`](/pkg/net/http#ServeContent),
[`net/http.ServeFile`](/pkg/net/http#ServeFile), and
[`net/http.ServeFS`](/pkg/net/http#ServeFS) to
remove Cache-Control, Content-Encoding, Etag, and Last-Modified headers
when serving an error. This behavior is controlled by
the [`httpservecontentkeepheaders` setting](/pkg/net/http#ServeContent).
Using `httpservecontentkeepheaders=1` restores the pre-Go 1.23 behavior.

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
For Go 1.23, `gotypesalias=1` will become the default.
This setting will be removed in a future release, Go 1.27 at the earliest.

Go 1.22 changed the default minimum TLS version supported by both servers
and clients to TLS 1.2. The default can be reverted to TLS 1.0 using the
[`tls10server` setting](/pkg/crypto/tls/#Config).

Go 1.22 changed the default TLS cipher suites used by clients and servers when
not explicitly configured, removing the cipher suites which used RSA based key
exchange. The default can be reverted using the [`tlsrsakex` setting](/pkg/crypto/tls/#Config).

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
[`disablethp` setting](/pkg/runtime#hdr-Environment_Variables).
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
setting](/pkg/runtime#hdr-Environment_Variables). These stack traces have
non-standard semantics, see setting documentation for details.

Go 1.22 added a new [`crypto/x509.Certificate`](/pkg/crypto/x509/#Certificate)
field, [`Policies`](/pkg/crypto/x509/#Certificate.Policies), which supports
certificate policy OIDs with components larger than 31 bits. By default this
field is only used during parsing, when it is populated with policy OIDs, but
not used during marshaling. It can be used to marshal these larger OIDs, instead
of the existing PolicyIdentifiers field, by using the
[`x509usepolicies` setting](/pkg/crypto/x509/#CreateCertificate).


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

Go 1.19 started sending EDNS0 additional headers on DNS requests.
This can reportedly break the DNS server provided on some routers,
such as CenturyLink Zyxel C3000Z.
This can be changed by the [`netedns0` setting](/pkg/net#hdr-Name_Resolution).
This setting is available in Go 1.21.12, Go 1.22.5, Go 1.23, and later.
There is no plan to remove this setting.

### Go 1.18

Go 1.18 removed support for SHA1 in most X.509 certificates,
controlled by the [`x509sha1` setting](/pkg/crypto/x509#InsecureAlgorithmError).
This setting was removed in Go 1.24.

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

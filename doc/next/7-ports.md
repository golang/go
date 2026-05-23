## Ports {#ports}

### Darwin {#darwin}

<!-- go.dev/issue/75836 -->
As [announced](go1.26#darwin) in the Go 1.26 release notes,
Go 1.27 requires macOS 13 Ventura or later;
support for previous versions has been discontinued.

### Linux {#linux}
<!-- go.dev/issue/76244 -->
On ppc64, the ABI has been migrated to ELFv2. This change
has no effect for those building and running pure Go
binaries.

On ppc64, external linking, cgo, and PIE binaries are now
supported. Using these features requires an ELFv2 compatible
runtime (libc, kernel, and all linked and loaded libraries).

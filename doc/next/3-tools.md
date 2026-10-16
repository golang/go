## Tools {#tools}

### Go command {#go-command}

<!-- go.dev/issue/26492 -->
The new `-static` build flag builds statically linked executables,
which do not depend on shared libraries at run time.
It replaces the combination of build tags, linker flags, and environment
settings that used to be needed, and that varied from system to system.
A program that uses cgo is linked against static versions of the C libraries
that it uses, and packages [net] and [os/user] use their pure Go
implementations. The flag is supported on DragonFly BSD, FreeBSD, Linux,
and NetBSD.

### Cgo {#cgo}

### Vet {#vet}

The new [`scannererr`](https://pkg.go.dev/golang.org/x/tools/go/analysis/passes/scannererr)
analyzer checks for failure to handle scanner errors after a loop
around [bufio.Scanner.Scan], which may cause scanning or I/O errors to
go unreported. <!-- /issue/17747/ -->

The [`sqlrowserr`](https://pkg.go.dev/golang.org/x/tools/go/analysis/passes/sqlrowserr)
analyzer performs a similar check for loops around [sql.Rows.Next],
so that iteration errors are correctly distinguished from a smaller result.

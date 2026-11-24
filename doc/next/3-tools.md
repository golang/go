## Tools {#tools}

### Go command {#go-command}

### Cgo {#cgo}

### Vet {#vet}

The new [`scannererr`](https://pkg.go.dev/golang.org/x/tools/go/analysis/passes/scannererr)
analyzer checks for failure to handle scanner errors after a loop
around [bufio.Scanner.Scan], which may cause scanning or I/O errors to
go unreported. <!-- /issue/17747/ -->

The [`sqlrowserr`](https://pkg.go.dev/golang.org/x/tools/go/analysis/passes/sqlrowserr)
analyzer performs a similar check for loops around [sql.Rows.Next],
so that iteration errors are correctly distinguished from a smaller result.

### Trace {#trace}

The "User-defined regions" page in `go tool trace` now accepts a regular
expression on region names. Its timeline shows matching regions together,
including regions from different tasks and regions without a task.
<!-- go.dev/issue/30895 -->

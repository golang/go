## Tools {#tools}

### Go command {#go-command}

<!-- go.dev/issue/81250 -->
The `GOFIPS140=certified` alias now selects v1.26.0 of the
[FIPS 140-3 Go Cryptographic Module](/doc/security/fips140), which obtained
[CMVP Certificate #5581](https://csrc.nist.gov/projects/cryptographic-module-validation-program/certificate/5581).

Support GOAMD64=local, like GOAMD64=vx,
where x is the value for the local micro-architecture.

### Cgo {#cgo}

### Vet {#vet}

The new [`scannererr`](https://pkg.go.dev/golang.org/x/tools/go/analysis/passes/scannererr)
analyzer checks for failure to handle scanner errors after a loop
around [bufio.Scanner.Scan], which may cause scanning or I/O errors to
go unreported. <!-- /issue/17747/ -->

The [`sqlrowserr`](https://pkg.go.dev/golang.org/x/tools/go/analysis/passes/sqlrowserr)
analyzer performs a similar check for loops around [sql.Rows.Next],
so that iteration errors are correctly distinguished from a smaller result.

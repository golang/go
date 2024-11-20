This directory holds snapshots of the crypto/internal/fips tree
that are being validated and certified for FIPS-140 use.
The file x.txt (for example, inprocess.txt, certified.txt)
defines the meaning of the FIPS version alias x, listing
the exact version to use.

The zip files are created by cmd/go/internal/fips/mkzip.go.
The fips140.sum file lists checksums for the zip files.
See the Makefile for recipes.

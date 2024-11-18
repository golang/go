// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fipsdeps

import (
	"internal/testenv"
	"strings"
	"testing"
)

// AllowedInternalPackages are internal packages that can be imported from the
// FIPS module. The API of these packages ends up locked for the lifetime of the
// validated module, which can be years.
//
// DO NOT add new packages here just to make the tests pass.
var AllowedInternalPackages = map[string]bool{
	// entropy.Depleted is the external passive entropy source, and sysrand.Read
	// is the actual (but uncredited!) random bytes source.
	"crypto/internal/entropy": true,
	"crypto/internal/sysrand": true,

	// impl.Register is how the packages expose their alternative
	// implementations to tests outside the module.
	"crypto/internal/impl": true,

	// randutil.MaybeReadByte is used in non-FIPS mode by GenerateKey functions.
	"crypto/internal/randutil": true,
}

func TestImports(t *testing.T) {
	cmd := testenv.Command(t, testenv.GoToolPath(t), "list", "-f", `{{$path := .ImportPath -}}
{{range .Imports -}}
{{$path}} {{.}}
{{end -}}
{{range .TestImports -}}
{{$path}} {{.}}
{{end -}}
{{range .XTestImports -}}
{{$path}} {{.}}
{{end -}}`, "crypto/internal/fips/...")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("go list: %v\n%s", err, out)
	}

	allPackages := make(map[string]bool)

	// importCheck is the set of packages that import crypto/internal/fips/check.
	importCheck := make(map[string]bool)

	for _, line := range strings.Split(string(out), "\n") {
		if line == "" {
			continue
		}
		pkg, importedPkg, _ := strings.Cut(line, " ")

		allPackages[pkg] = true

		if importedPkg == "crypto/internal/fips/check" {
			importCheck[pkg] = true
		}

		// Ensure we don't import any unexpected internal package from the FIPS
		// module, since we can't change the module source after it starts
		// validation. This locks in the API of otherwise internal packages.
		if importedPkg == "crypto/internal/fips" ||
			strings.HasPrefix(importedPkg, "crypto/internal/fips/") ||
			strings.HasPrefix(importedPkg, "crypto/internal/fipsdeps/") {
			continue
		}
		if AllowedInternalPackages[importedPkg] {
			continue
		}
		if strings.Contains(importedPkg, "internal") {
			t.Errorf("unexpected import of internal package: %s -> %s", pkg, importedPkg)
		}
	}

	// Ensure that all packages except check and check's dependencies import check.
	for pkg := range allPackages {
		switch pkg {
		case "crypto/internal/fips/check":
		case "crypto/internal/fips":
		case "crypto/internal/fips/alias":
		case "crypto/internal/fips/subtle":
		case "crypto/internal/fips/hmac":
		case "crypto/internal/fips/sha3":
		case "crypto/internal/fips/sha256":
		case "crypto/internal/fips/sha512":
		default:
			if !importCheck[pkg] {
				t.Errorf("package %s does not import crypto/internal/fips/check", pkg)
			}
		}
	}
}

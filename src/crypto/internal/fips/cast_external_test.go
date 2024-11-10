// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fips_test

import (
	"crypto/internal/fips"
	"fmt"
	"internal/testenv"
	"strings"
	"testing"

	// Import packages that define CASTs to test them.
	_ "crypto/internal/fips/aes"
	_ "crypto/internal/fips/aes/gcm"
	_ "crypto/internal/fips/drbg"
	_ "crypto/internal/fips/hkdf"
	_ "crypto/internal/fips/hmac"
	_ "crypto/internal/fips/sha256"
	_ "crypto/internal/fips/sha3"
	_ "crypto/internal/fips/sha512"
	_ "crypto/internal/fips/tls12"
	_ "crypto/internal/fips/tls13"
)

func TestCAST(t *testing.T) {
	if len(fips.AllCASTs) == 0 {
		t.Errorf("no CASTs to test")
	}

	if fips.Enabled {
		for _, name := range fips.AllCASTs {
			t.Logf("CAST %s completed successfully", name)
		}
	}

	t.Run("SimulateFailures", func(t *testing.T) {
		testenv.MustHaveExec(t)
		for _, name := range fips.AllCASTs {
			t.Run(name, func(t *testing.T) {
				t.Parallel()
				cmd := testenv.Command(t, testenv.Executable(t), "-test.run=TestCAST", "-test.v")
				cmd = testenv.CleanCmdEnv(cmd)
				cmd.Env = append(cmd.Env, fmt.Sprintf("GODEBUG=failfipscast=%s,fips140=on", name))
				out, err := cmd.CombinedOutput()
				if err == nil {
					t.Error(err)
				} else {
					t.Logf("CAST %s failed and caused the program to exit", name)
					t.Logf("%s", out)
				}
				if strings.Contains(string(out), "completed successfully") {
					t.Errorf("CAST %s failure did not stop the program", name)
				}
			})
		}
	})
}

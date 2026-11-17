// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"internal/testenv"
	"os"
	"path/filepath"
	"testing"

	"cmd/compile/internal/syntax"
)

// Run the generated methods, so a runtime helper test cannot stand in for
// the compiler's method selection or its scalar fallback after a decline.
func TestJSONProfileEjectedBehavior(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	for _, name := range []string{
		"json_string_method_input_test.lgo",
		"json_string_method_wrong_sig_test.lgo",
		"json_string_method_stream_test.lgo",
		"json_string_method_decline_test.lgo",
		"json_string_pointer_null_test.lgo",
		"json_marshal_v2_test.lgo",
		"json_static_encoder_methods_test.lgo",
		"json_unsupported_kinds_test.lgo",
	} {
		t.Run(name, func(t *testing.T) {
			path := filepath.Join(runLogosDir(t), name)
			skipIfBuildTagsExclude(t, path)
			src, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			base := syntax.NewFileBase(path)
			file := ejectParse(t, base, src, syntax.LogosMode|syntax.CheckBranches)
			pkg, info := ejectCheck(t, file)
			edits := ejectLower(t, pkg, info, file, src)
			ejected, err := ApplyEdits(src, base, edits)
			if err != nil {
				t.Fatalf("ApplyEdits: %v", err)
			}
			work := t.TempDir()
			if err := os.WriteFile(filepath.Join(work, "go.mod"), []byte("module jsonprofile\n\ngo 1.27\n"), 0o644); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(work, "profile_test.go"), ejected, 0o644); err != nil {
				t.Fatal(err)
			}
			cmd := testenv.Command(t, testenv.GoToolPath(t), "test", "-count=1", ".")
			cmd.Dir = work
			cmd.Env = append(os.Environ(), "GOPROXY=off", "GOTOOLCHAIN=local")
			if out, err := cmd.CombinedOutput(); err != nil {
				t.Fatalf("ejected test: %v\n%s", err, out)
			}
		})
	}
}

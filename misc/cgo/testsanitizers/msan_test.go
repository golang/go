// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sanitizers_test

import (
	"strings"
	"testing"
)

func TestMSAN(t *testing.T) {
	t.Parallel()
	requireOvercommit(t)
	config := configure("memory")
	config.skipIfCSanitizerBroken(t)

	mustRun(t, config.goCmd("build", "std"))

	cases := []struct {
		src     string
		wantErr bool
	}{
		{src: "msan.go"},
		{src: "msan2.go"},
		{src: "msan2_cmsan.go"},
		{src: "msan3.go"},
		{src: "msan4.go"},
		{src: "msan5.go"},
		{src: "msan6.go"},
		{src: "msan_fail.go", wantErr: true},
	}
	for _, tc := range cases {
		tc := tc
		name := strings.TrimSuffix(tc.src, ".go")
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			dir := newTempDir(t)
			defer dir.RemoveAll(t)

			outPath := dir.Join(name)
			mustRun(t, config.goCmd("build", "-o", outPath, srcPath(tc.src)))

			cmd := hangProneCmd(outPath)
			if tc.wantErr {
				out, err := cmd.CombinedOutput()
				if err != nil {
					return
				}
				t.Fatalf("%#q exited without error; want MSAN failure\n%s", strings.Join(cmd.Args, " "), out)
			}
			mustRun(t, cmd)
		})
	}
}

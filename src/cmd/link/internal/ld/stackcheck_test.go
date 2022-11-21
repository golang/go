// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"fmt"
	"internal/testenv"
	"os"
	"regexp"
	"strconv"
	"testing"
)

// See also $GOROOT/test/nosplit.go for multi-platform edge case tests.

func TestStackCheckOutput(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	t.Parallel()

	cmd := testenv.Command(t, testenv.GoToolPath(t), "build", "-o", os.DevNull, "./testdata/stackcheck")
	// The rules for computing frame sizes on all of the
	// architectures are complicated, so just do this on amd64.
	cmd.Env = append(os.Environ(), "GOARCH=amd64", "GOOS=linux")
	outB, err := cmd.CombinedOutput()

	if err == nil {
		t.Fatalf("expected link to fail")
	}
	out := string(outB)

	t.Logf("linker output:\n%s", out)

	// Get expected limit.
	limitRe := regexp.MustCompile(`nosplit stack over (\d+) byte limit`)
	m := limitRe.FindStringSubmatch(out)
	if m == nil {
		t.Fatalf("no overflow errors in output")
	}
	limit, _ := strconv.Atoi(m[1])

	wantMap := map[string]string{
		"main.startSelf": fmt.Sprintf(
			`main.startSelf<0>
    grows 1008 bytes
    %d bytes over limit
`, 1008-limit),
		"main.startChain": fmt.Sprintf(
			`main.startChain<0>
    grows 32 bytes, calls main.chain0<0>
        grows 48 bytes, calls main.chainEnd<0>
            grows 1008 bytes
            %d bytes over limit
    grows 32 bytes, calls main.chain2<0>
        grows 80 bytes, calls main.chainEnd<0>
            grows 1008 bytes
            %d bytes over limit
`, 32+48+1008-limit, 32+80+1008-limit),
		"main.startRec": `main.startRec<0>
    grows 8 bytes, calls main.startRec0<0>
        grows 8 bytes, calls main.startRec<0>
        infinite cycle
`,
	}

	// Parse stanzas
	stanza := regexp.MustCompile(`^(.*): nosplit stack over \d+ byte limit\n(.*\n(?: .*\n)*)`)
	// Strip comments from cmd/go
	out = regexp.MustCompile(`(?m)^#.*\n`).ReplaceAllString(out, "")
	for len(out) > 0 {
		m := stanza.FindStringSubmatch(out)
		if m == nil {
			t.Fatalf("unexpected output:\n%s", out)
		}
		out = out[len(m[0]):]
		fn := m[1]
		got := m[2]

		want, ok := wantMap[fn]
		if !ok {
			t.Errorf("unexpected function: %s", fn)
		} else if want != got {
			t.Errorf("want:\n%sgot:\n%s", want, got)
		}
	}
}

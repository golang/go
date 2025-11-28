// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testtrace

import (
	"bytes"
	"fmt"
	"internal/testenv"
	"internal/trace/raw"
	"internal/trace/version"
	"io"
	"os"
	"strings"
	"testing"
)

// Dump saves the trace to a file or the test log.
func Dump(t *testing.T, testName string, traceBytes []byte, forceToFile bool) {
	onBuilder := testenv.Builder() != ""
	onOldBuilder := !strings.Contains(testenv.Builder(), "gotip") && !strings.Contains(testenv.Builder(), "go1")

	if onBuilder && !forceToFile {
		// Dump directly to the test log on the builder, since this
		// data is critical for debugging and this is the only way
		// we can currently make sure it's retained.
		s := dumpTraceToText(t, traceBytes)
		if onOldBuilder && len(s) > 1<<20+512<<10 {
			// The old build infrastructure truncates logs at ~2 MiB.
			// Let's assume we're the only failure and give ourselves
			// up to 1.5 MiB to dump the trace.
			//
			// TODO(mknyszek): Remove this when we've migrated off of
			// the old infrastructure.
			t.Logf("text trace too large to dump (%d bytes)", len(s))
		} else {
			t.Log(s)
			t.Log("Convert this to a raw trace with `go test internal/trace/testtrace -convert in.tracetxt -out out.trace`")
		}
	} else {
		// We asked to dump the trace or failed. Write the trace to a file.
		t.Logf("wrote trace to file: %s", dumpTraceToFile(t, testName, traceBytes))
	}
}

func dumpTraceToText(t *testing.T, b []byte) string {
	t.Helper()

	br, err := raw.NewReader(bytes.NewReader(b))
	if err != nil {
		t.Fatalf("dumping trace: %v", err)
	}
	var sb strings.Builder
	tw, err := raw.NewTextWriter(&sb, version.Current)
	if err != nil {
		t.Fatalf("dumping trace: %v", err)
	}
	for {
		ev, err := br.ReadEvent()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("dumping trace: %v", err)
		}
		if err := tw.WriteEvent(ev); err != nil {
			t.Fatalf("dumping trace: %v", err)
		}
	}
	return sb.String()
}

func dumpTraceToFile(t *testing.T, testName string, b []byte) string {
	t.Helper()

	name := fmt.Sprintf("%s.trace.", testName)
	f, err := os.CreateTemp(t.ArtifactDir(), name)
	if err != nil {
		t.Fatalf("creating temp file: %v", err)
	}
	defer f.Close()
	if _, err := io.Copy(f, bytes.NewReader(b)); err != nil {
		t.Fatalf("writing trace dump to %q: %v", f.Name(), err)
	}
	return f.Name()
}

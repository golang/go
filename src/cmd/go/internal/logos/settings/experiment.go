// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package settings

import "strings"

const (
	arenaExperiment            = "logosarena"
	preemptibleLoopsExperiment = "preemptibleloops"
)

func WithArena(raw string) string {
	return appendExperiment(raw, arenaExperiment)
}

func PreemptibleDefault(raw, startup string, has, logos bool, goarch string) (wantRaw string, apply bool) {
	mentioned, _ := preemptibleUserState(startup)
	want := logos && goarch == "wasm" && !mentioned
	switch {
	case want && !has:
		return appendExperiment(raw, preemptibleLoopsExperiment), true
	case !want && has && !mentioned:
		return stripExperiment(raw, preemptibleLoopsExperiment), true
	}
	return "", false
}

func DropLogosExperiments(raw string) string {
	var kept []string
	for f := range strings.SplitSeq(raw, ",") {
		if f != "" && !strings.HasPrefix(f, "logos") && !strings.HasPrefix(f, "nologos") {
			kept = append(kept, f)
		}
	}
	return strings.Join(kept, ",")
}

func appendExperiment(raw, exp string) string {
	if raw == "" {
		return exp
	}
	return raw + "," + exp
}

func stripExperiment(raw, exp string) string {
	var kept []string
	for f := range strings.SplitSeq(raw, ",") {
		if f != exp {
			kept = append(kept, f)
		}
	}
	return strings.Join(kept, ",")
}

func preemptibleUserState(raw string) (mentioned, enabled bool) {
	for f := range strings.SplitSeq(raw, ",") {
		switch f {
		case preemptibleLoopsExperiment:
			mentioned, enabled = true, true
		case "no" + preemptibleLoopsExperiment, "none":
			mentioned, enabled = true, false
		}
	}
	return mentioned, enabled
}

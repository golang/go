// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package buildcfg

import (
	"fmt"
	"strings"
)

// GOGOARCHTags returns the architecture feature tags for a GOARCH and its
// GO$GOARCH setting. It does not read or change the process configuration.
// Callers selecting another target use the same implications as ToolTags.
func GOGOARCHTags(goarch, level string) ([]string, error) {
	return gogoarchTagsFor(goarch, level, true)
}

// The ordinary tool-tag path preserves raw GO386 values so the compiler can
// report its existing unsupported-setting diagnostic. Candidate selection has
// no such later validation boundary and must reject invalid settings here.
func gogoarchTagsFor(goarch, level string, validate386 bool) ([]string, error) {
	switch goarch {
	case "386":
		if !validate386 || level == "sse2" || level == "softfloat" {
			return []string{goarch + "." + level}, nil
		}
	case "amd64":
		if level == "v1" || level == "v2" || level == "v3" || level == "v4" {
			var tags []string
			for i := 1; i <= int(level[1]-'0'); i++ {
				tags = append(tags, fmt.Sprintf("%s.v%d", goarch, i))
			}
			return tags, nil
		}
	case "arm":
		version, float, _ := strings.Cut(level, ",")
		if (version == "5" || version == "6" || version == "7") && (float == "" || float == "softfloat" || float == "hardfloat") {
			var tags []string
			for i := 5; i <= int(version[0]-'0'); i++ {
				tags = append(tags, fmt.Sprintf("%s.%d", goarch, i))
			}
			return tags, nil
		}
	case "arm64":
		features, err := ParseGoarm64(level)
		if err != nil {
			return nil, err
		}
		var tags []string
		major := int(features.Version[1] - '0')
		minor := int(features.Version[3] - '0')
		for i := 0; i <= minor; i++ {
			tags = append(tags, fmt.Sprintf("%s.v%d.%d", goarch, major, i))
		}
		// ARM64 v9.x includes v8.(x+5), capped at v8.9.
		if major == 9 {
			for i := 0; i <= minor+5 && i <= 9; i++ {
				tags = append(tags, fmt.Sprintf("%s.v8.%d", goarch, i))
			}
		}
		return tags, nil
	case "mips", "mipsle", "mips64", "mips64le":
		if level == "hardfloat" || level == "softfloat" {
			return []string{goarch + "." + level}, nil
		}
	case "ppc64", "ppc64le":
		if level == "power8" || level == "power9" || level == "power10" {
			last := 10
			if level != "power10" {
				last = int(level[5] - '0')
			}
			var tags []string
			for i := 8; i <= last; i++ {
				tags = append(tags, fmt.Sprintf("%s.power%d", goarch, i))
			}
			return tags, nil
		}
	case "riscv64":
		if level == "rva20u64" || level == "rva22u64" || level == "rva23u64" {
			tags := []string{goarch + ".rva20u64"}
			if level != "rva20u64" {
				tags = append(tags, goarch+".rva22u64")
			}
			if level == "rva23u64" {
				tags = append(tags, goarch+".rva23u64")
			}
			return tags, nil
		}
	case "wasm":
		for feature := range strings.SplitSeq(level, ",") {
			if feature != "" && feature != "satconv" && feature != "signext" {
				return nil, fmt.Errorf("invalid GOWASM: no such feature %q", feature)
			}
		}
		return []string{goarch + ".satconv", goarch + ".signext"}, nil
	default:
		if level == "" {
			return nil, nil
		}
	}
	return nil, fmt.Errorf("invalid %s architecture feature level %q", goarch, level)
}

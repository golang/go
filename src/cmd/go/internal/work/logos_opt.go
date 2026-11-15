// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package work

import (
	"os"
	"strings"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/logos/settings"
)

func logosAddOptFlag(cmd *base.Command) {
	cmd.Flag.StringVar(&settings.CommandLine.Opt, "O", "", "")
}

func logosApplyOpt() {
	s := settings.Current()
	if logosSettingsErr != nil {
		base.Fatalf("%v", logosSettingsErr)
	}
	forcedGcflags = append(forcedGcflags, s.Gcflags...)
	logosApplyLoopStride()
	if s.GOAMD64 != "" {
		logosSetArchLevel("GOAMD64", s.GOAMD64)
	}
	if s.GOARM64 != "" {
		logosSetArchLevel("GOARM64", s.GOARM64)
	}
}

var logosTargetArchEnv []string

func LogosTargetArchEnv() []string {
	if !cfg.Logos {
		return nil
	}
	return settings.Current().TargetArchEnv(cfg.Goarch)
}

func logosSetArchLevel(key, val string) {
	if val == "" {
		return
	}
	if key == "GOAMD64" {
		cfg.GOAMD64 = val
	} else {
		cfg.GOARM64 = val
	}
	os.Setenv(key, val)
	cfgChangedEnv = append(cfgChangedEnv, key+"="+val)
	for i, entry := range logosTargetArchEnv {
		if strings.HasPrefix(entry, key+"=") {
			logosTargetArchEnv[i] = key + "=" + val
			return
		}
	}
	logosTargetArchEnv = append(logosTargetArchEnv, key+"="+val)
}

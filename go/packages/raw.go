// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package packages

import (
	"context"
	"fmt"
	"os"
)

// This file contains the structs needed at the seam between the packages
// loader and the underlying build tool

// rawConfig specifies details about what raw package information is needed
// and how the underlying build tool should load package data.
type rawConfig struct {
	Context context.Context
	Dir     string
	Env     []string
	Export  bool
	Tests   bool
	Deps    bool
}

func newRawConfig(cfg *Config) *rawConfig {
	rawCfg := &rawConfig{
		Context: cfg.Context,
		Dir:     cfg.Dir,
		Env:     cfg.Env,
		Export:  cfg.Mode > LoadImports && cfg.Mode < LoadAllSyntax,
		Tests:   cfg.Tests,
		Deps:    cfg.Mode >= LoadImports,
	}
	if rawCfg.Env == nil {
		rawCfg.Env = os.Environ()
	}
	return rawCfg
}

func (cfg *rawConfig) Flags() []string {
	return []string{
		fmt.Sprintf("-test=%t", cfg.Tests),
		fmt.Sprintf("-export=%t", cfg.Export),
		fmt.Sprintf("-deps=%t", cfg.Deps),
	}
}

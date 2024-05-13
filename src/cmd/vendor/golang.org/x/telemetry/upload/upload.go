// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package upload

import (
	"log"

	"golang.org/x/telemetry/internal/upload"
)

// TODO(rfindley): remove, in favor of all callers using Start.

// A RunConfig controls the behavior of Run.
// The zero value RunConfig is the default behavior; fields may be set to
// override various reporting and uploading choices.
type RunConfig = upload.RunConfig

// Run generates and uploads reports, as allowed by the mode file.
func Run(config RunConfig) error {
	defer func() {
		if err := recover(); err != nil {
			log.Printf("upload recover: %v", err)
		}
	}()
	uploader, err := upload.NewUploader(config)
	if err != nil {
		return err
	}
	defer uploader.Close()
	return uploader.Run()
}

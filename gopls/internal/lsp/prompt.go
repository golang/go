// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/telemetry"
	"golang.org/x/tools/internal/event"
)

// promptTimeout is the amount of time we wait for an ongoing prompt before
// prompting again. This gives the user time to reply. However, at some point
// we must assume that the client is not displaying the prompt, the user is
// ignoring it, or the prompt has been disrupted in some way (e.g. by a gopls
// crash).
const promptTimeout = 24 * time.Hour

// The following constants are used for testing telemetry integration.
const (
	TelemetryPromptWorkTitle    = "Checking telemetry prompt"     // progress notification title, for awaiting in tests
	GoplsConfigDirEnvvar        = "GOPLS_CONFIG_DIR"              // overridden for testing
	FakeTelemetryModefileEnvvar = "GOPLS_FAKE_TELEMETRY_MODEFILE" // overridden for testing
	TelemetryYes                = "Yes, I'd like to help."
	TelemetryNo                 = "No, thanks."
)

// getenv returns the effective environment variable value for the provided
// key, looking up the key in the session environment before falling back on
// the process environment.
func (s *Server) getenv(key string) string {
	if v, ok := s.Options().Env[key]; ok {
		return v
	}
	return os.Getenv(key)
}

// configDir returns the root of the gopls configuration dir. By default this
// is os.UserConfigDir/gopls, but it may be overridden for tests.
func (s *Server) configDir() (string, error) {
	if d := s.getenv(GoplsConfigDirEnvvar); d != "" {
		return d, nil
	}
	userDir, err := os.UserConfigDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(userDir, "gopls"), nil
}

// telemetryMode returns the current effective telemetry mode.
// By default this is x/telemetry.Mode(), but it may be overridden for tests.
func (s *Server) telemetryMode() string {
	if fake := s.getenv(FakeTelemetryModefileEnvvar); fake != "" {
		if data, err := os.ReadFile(fake); err == nil {
			return string(data)
		}
		return "off"
	}
	return telemetry.Mode()
}

// setTelemetryMode sets the current telemetry mode.
// By default this calls x/telemetry.SetMode, but it may be overridden for
// tests.
func (s *Server) setTelemetryMode(mode string) error {
	if fake := s.getenv(FakeTelemetryModefileEnvvar); fake != "" {
		return os.WriteFile(fake, []byte(mode), 0666)
	}
	return telemetry.SetMode(mode)
}

// maybePromptForTelemetry checks for the right conditions, and then prompts
// the user to ask if they want to enable Go telemetry uploading. If the user
// responds 'Yes', the telemetry mode is set to "on".
//
// The actual conditions for prompting are defensive, erring on the side of not
// prompting.
// If enabled is false, this will not prompt the user in any condition,
// but will send work progress reports to help testing.
func (s *Server) maybePromptForTelemetry(ctx context.Context, enabled bool) {
	if s.Options().VerboseWorkDoneProgress {
		work := s.progress.Start(ctx, TelemetryPromptWorkTitle, "Checking if gopls should prompt about telemetry...", nil, nil)
		defer work.End(ctx, "Done.")
	}

	if !enabled { // check this after the work progress message for testing.
		return // prompt is disabled
	}

	if s.telemetryMode() == "on" {
		// Telemetry is already on -- nothing to ask about.
		return
	}

	errorf := func(format string, args ...any) {
		err := fmt.Errorf(format, args...)
		event.Error(ctx, "telemetry prompt failed", err)
	}

	// Only prompt if we can read/write the prompt config file.
	configDir, err := s.configDir()
	if err != nil {
		errorf("unable to determine config dir: %v", err)
		return
	}

	var (
		promptDir  = filepath.Join(configDir, "prompt")    // prompt configuration directory
		promptFile = filepath.Join(promptDir, "telemetry") // telemetry prompt file
	)

	// prompt states, to be written to the prompt file
	const (
		pYes     = "yes"     // user said yes
		pNo      = "no"      // user said no
		pPending = "pending" // current prompt is still pending
		pFailed  = "failed"  // prompt was asked but failed
	)
	validStates := map[string]bool{
		pYes:     true,
		pNo:      true,
		pPending: true,
		pFailed:  true,
	}

	// parse the current prompt file
	var (
		state    string
		attempts = 0 // number of times we've asked already
	)
	if content, err := os.ReadFile(promptFile); err == nil {
		if _, err := fmt.Sscanf(string(content), "%s %d", &state, &attempts); err == nil && validStates[state] {
			if state == pYes || state == pNo {
				// Prompt has been answered. Nothing to do.
				return
			}
		} else {
			state, attempts = "", 0
			errorf("malformed prompt result %q", string(content))
		}
	} else if !os.IsNotExist(err) {
		errorf("reading prompt file: %v", err)
		// Something went wrong. Since we don't know how many times we've asked the
		// prompt, err on the side of not spamming.
		return
	}

	if attempts >= 5 {
		// We've tried asking enough; give up.
		return
	}
	if attempts == 0 {
		// First time asking the prompt; we may need to make the prompt dir.
		if err := os.MkdirAll(promptDir, 0777); err != nil {
			errorf("creating prompt dir: %v", err)
			return
		}
	}

	// Acquire the lock and write "pending" to the prompt file before actually
	// prompting.
	//
	// This ensures that the prompt file is writeable, and that we increment the
	// attempt counter before we prompt, so that we don't end up in a failure
	// mode where we keep prompting and then failing to record the response.

	release, ok, err := acquireLockFile(promptFile)
	if err != nil {
		errorf("acquiring prompt: %v", err)
		return
	}
	if !ok {
		// Another prompt is currently pending.
		return
	}
	defer release()

	attempts++

	pendingContent := []byte(fmt.Sprintf("%s %d", pPending, attempts))
	if err := os.WriteFile(promptFile, pendingContent, 0666); err != nil {
		errorf("writing pending state: %v", err)
		return
	}

	var prompt = `Go telemetry helps us improve Go by periodically sending anonymous metrics and crash reports to the Go team. Learn more at https://telemetry.go.dev/privacy.

Would you like to enable Go telemetry?
`
	if s.Options().LinkifyShowMessage {
		prompt = `Go telemetry helps us improve Go by periodically sending anonymous metrics and crash reports to the Go team. Learn more at [telemetry.go.dev/privacy](https://telemetry.go.dev/privacy).

Would you like to enable Go telemetry?
`
	}
	// TODO(rfindley): investigate a "tell me more" action in combination with ShowDocument.
	params := &protocol.ShowMessageRequestParams{
		Type:    protocol.Info,
		Message: prompt,
		Actions: []protocol.MessageActionItem{
			{Title: TelemetryYes},
			{Title: TelemetryNo},
		},
	}

	item, err := s.client.ShowMessageRequest(ctx, params)
	if err != nil {
		errorf("ShowMessageRequest failed: %v", err)
		// Defensive: ensure item == nil for the logic below.
		item = nil
	}

	message := func(typ protocol.MessageType, msg string) {
		if err := s.client.ShowMessage(ctx, &protocol.ShowMessageParams{
			Type:    typ,
			Message: msg,
		}); err != nil {
			errorf("ShowMessage(unrecognize) failed: %v", err)
		}
	}

	result := pFailed
	if item == nil {
		// e.g. dialog was dismissed
		errorf("no response")
	} else {
		// Response matches MessageActionItem.Title.
		switch item.Title {
		case TelemetryYes:
			result = pYes
			if err := s.setTelemetryMode("on"); err == nil {
				message(protocol.Info, telemetryOnMessage(s.Options().LinkifyShowMessage))
			} else {
				errorf("enabling telemetry failed: %v", err)
				msg := fmt.Sprintf("Failed to enable Go telemetry: %v\nTo enable telemetry manually, please run `go run golang.org/x/telemetry/cmd/gotelemetry@latest on`", err)
				message(protocol.Error, msg)
			}

		case TelemetryNo:
			result = pNo
		default:
			errorf("unrecognized response %q", item.Title)
			message(protocol.Error, fmt.Sprintf("Unrecognized response %q", item.Title))
		}
	}
	resultContent := []byte(fmt.Sprintf("%s %d", result, attempts))
	if err := os.WriteFile(promptFile, resultContent, 0666); err != nil {
		errorf("error writing result state to prompt file: %v", err)
	}
}

func telemetryOnMessage(linkify bool) string {
	format := `Thank you. Telemetry uploading is now enabled.

To disable telemetry uploading, run %s.
`
	var runCmd = "`go run golang.org/x/telemetry/cmd/gotelemetry@latest off`"
	if linkify {
		runCmd = "[gotelemetry off](https://golang.org/x/telemetry/cmd/gotelemetry)"
	}
	return fmt.Sprintf(format, runCmd)
}

// acquireLockFile attempts to "acquire a lock" for writing to path.
//
// This is achieved by creating an exclusive lock file at <path>.lock. Lock
// files expire after a period, at which point acquireLockFile will remove and
// recreate the lock file.
//
// acquireLockFile fails if path is in a directory that doesn't exist.
func acquireLockFile(path string) (func(), bool, error) {
	lockpath := path + ".lock"
	fi, err := os.Stat(lockpath)
	if err == nil {
		if time.Since(fi.ModTime()) > promptTimeout {
			_ = os.Remove(lockpath) // ignore error
		} else {
			return nil, false, nil
		}
	} else if !os.IsNotExist(err) {
		return nil, false, fmt.Errorf("statting lockfile: %v", err)
	}

	f, err := os.OpenFile(lockpath, os.O_CREATE|os.O_EXCL, 0666)
	if err != nil {
		if os.IsExist(err) {
			return nil, false, nil
		}
		return nil, false, fmt.Errorf("creating lockfile: %v", err)
	}
	fi, err = f.Stat()
	if err != nil {
		return nil, false, err
	}
	release := func() {
		_ = f.Close() // ignore error
		fi2, err := os.Stat(lockpath)
		if err == nil && os.SameFile(fi, fi2) {
			// Only clean up the lockfile if it's the same file we created.
			// Otherwise, our lock has expired and something else has the lock.
			//
			// There's a race here, in that the file could have changed since the
			// stat above; but given that we've already waited 24h this is extremely
			// unlikely, and acceptable.
			_ = os.Remove(lockpath)
		}
	}
	return release, true, nil
}

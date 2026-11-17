# Overlay-aware inactive ejection: component evidence

Slice M24-A-S3; REDTEAM #441. The incoming frozen unit and script are unchanged 8afec955 bytes. This gate binds all 18 merged ejection package files by SHA-256 and Git blob identity in source-manifest.json. It uses the immutable 45c29ff2 installed compiler and read-only candidate tool links. No product source or frozen assertion was edited.

## RED

The unit-only overlay adds the incoming test to unchanged 45c29ff2. It fails because the physical census includes removed.lgo after the overlay deletes it; target matching reports deleted in overlay. This precedes the final vector assertion anticipated by the initial external log classifier. The actual semantic mechanism is recorded explicitly in baseline-verdict.json.

The baseline script also fails: the physical census retains ignored.lgo despite effective go:build ignore, and target selection reports no valid build context. This run reuses the hash-verified existing 45c29ff2 cmd/go race binary; no cmd/go binary was rebuilt. The frozen script is read from the candidate working directory while the command implementation and GOROOT remain at 45c29ff2.

## GREEN and probe

The merged-source TestInactiveCensusHonorsOverlay race gate passed in 2.15s. One external overlay changes only fsys.ReadFile(path) to os.ReadFile(path) in inactive.go. That mutant compiles and is caught because the physical read cannot open overlay-inserted inserted.lgo. The frozen test and all other source bytes remain unchanged.

A fresh no-overlay race binary passes the same targeted test after the probe (2.064s). Normal go vet cmd/go/internal/logos/eject passes (1.431s). Both GREEN executions have no skips, race report or panic. Compilation and cache stay on disk; runtime source temporaries use a unique real /home/agent directory. All commands are bounded and use one worker with a 512 MiB limit.

## Remaining acceptance

The new-source installed native/eject script has deliberately not run. It must run after the final clean make against the merged installed command and compiler. Component GREEN does not replace that gate.

Keep the raw logs, exact exit records, frozen inputs, overlay mappings, source copies and checksums. Do not commit the four .test binaries. The immutable 45c29ff2 and candidate source trees were not changed by this validation.

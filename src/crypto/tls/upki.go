// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"crypto/internal/upki"
	"crypto/x509"
	"errors"
	"fmt"
	"internal/godebug"
	"os"
	"path/filepath"
	"runtime"
)

// tlsupki is a GODEBUG setting that can be set to 0 to disable UPKI
// revocation checking for configurations that rely on platform-default
// cache discovery.
var tlsupki = godebug.New("tlsupki")

// UPKIRevocationConfig controls local, offline UPKI revocation checking of
// server certificates.
//
// It doesn't configure downloading or updating revocation data.
type UPKIRevocationConfig struct {
	// Disabled disables UPKI revocation checking, including automatic
	// discovery of a platform-default cache.
	Disabled bool

	// Directory is a C2SP upki-revocation cache directory. It contains
	// index.bin directly.
	//
	// If Directory is empty, platform-default directories are searched for
	// index.bin in user-before-system order. Discovery skips missing paths,
	// non-directory path components, and paths the process lacks permission
	// to open.
	Directory string

	// MissingData controls the result when a lookup needs an index but its
	// path is missing, a parent is not a directory, or access is denied.
	// By default, the connection proceeds if automatic discovery finds no
	// available index. [UPKIRevocationPolicyReject] requires an available index.
	// Errors reading or parsing an opened index or its filters are fatal.
	MissingData UPKIRevocationPolicy

	// NotCovered controls the result when the revocation data cannot
	// determine a certificate's status, including when covering filters have
	// no information about the certificate's issuer. By default the connection
	// proceeds. [UPKIRevocationPolicyReject] produces an error if there isn't
	// sufficient coverage to make a revocation determination.
	NotCovered UPKIRevocationPolicy

	// MissingSCTs controls the result when a certificate contains no embedded
	// Signed Certificate Timestamps. A malformed embedded SCT extension is an
	// error if a cache is explicitly configured or found during discovery.
	// By default a certificate without SCTs proceeds without error.
	// [UPKIRevocationPolicyReject] produces an error if the SCTs required for
	// a revocation determination aren't present.
	MissingSCTs UPKIRevocationPolicy
}

// ErrUPKIRevoked is returned as a wrapped [CertificateVerificationError]
// when a certificate is determined to be revoked.
var ErrUPKIRevoked = errors.New("certificate is revoked according to UPKI")

// UPKIRevocationPolicy controls the result of an inconclusive UPKI revocation
// check.
type UPKIRevocationPolicy uint8

const (
	// UPKIRevocationPolicyDefault uses the default for the corresponding
	// UPKIRevocationConfig field.
	UPKIRevocationPolicyDefault UPKIRevocationPolicy = iota

	// UPKIRevocationPolicyAllow allows the connection to proceed when the
	// revocation check is inconclusive for the corresponding reason.
	UPKIRevocationPolicyAllow

	// UPKIRevocationPolicyReject rejects the connection when the revocation
	// check is inconclusive for the corresponding reason.
	UPKIRevocationPolicyReject
)

func (c *Config) checkUPKIRevocation(chains [][]*x509.Certificate) error {
	// ECH rejection can produce verified chains for the public name even
	// when InsecureSkipVerify is set.
	if len(chains) == 0 {
		return nil
	}

	cfg := c.UPKIRevocation
	missingDataRejects, err := upkiPolicyReject(cfg.MissingData, cfg.Directory != "")
	if err != nil {
		return err
	}
	notCoveredRejects, err := upkiPolicyReject(cfg.NotCovered, false)
	if err != nil {
		return err
	}
	missingSCTsRejects, err := upkiPolicyReject(cfg.MissingSCTs, false)
	if err != nil {
		return err
	}

	if cfg.Disabled {
		return nil
	}
	// GODEBUG=tlsupki=0 turns off the default-on checking. A Config that
	// names a cache directory explicitly keeps checking it.
	if cfg.Directory == "" && tlsupki.Value() == "0" {
		tlsupki.IncNonDefault()

		return nil
	}

	// During discovery, leave malformed SCTs for upki.Check to report if
	// we find an index. MissingSCTs applies before any cache access.
	scts, err := upki.ExtractSCTs(chains[0][0])
	if err != nil && cfg.Directory != "" {
		return fmt.Errorf("UPKI revocation check failed: extracting SCTs: %w", err)
	}
	if err == nil && len(scts) == 0 {
		if missingSCTsRejects {
			return errors.New("UPKI revocation check requires embedded SCTs")
		}

		return nil
	}

	directories := []string{cfg.Directory}
	if cfg.Directory == "" {
		directories = defaultUPKIRevocationDirectories()
	}
	dir, index, err := openUPKICacheIndex(directories)
	if err != nil {
		if errors.Is(err, upki.ErrNoData) && !missingDataRejects {
			return nil
		}
		return err
	}

	result, err := upki.Check(dir, index, chains)
	if err != nil {
		if errors.Is(err, upki.ErrNoData) && cfg.MissingData == UPKIRevocationPolicyAllow {
			return nil
		}

		return fmt.Errorf("UPKI revocation check failed: %w", err)
	}

	switch result {
	case upki.Revoked:
		return ErrUPKIRevoked
	case upki.NotRevoked:
		return nil
	case upki.NotCovered:
		if notCoveredRejects {
			return errors.New("certificate is not covered by UPKI revocation data")
		}

		return nil
	default:
		return fmt.Errorf("UPKI revocation check returned invalid result %d", result)
	}
}

func upkiPolicyReject(policy UPKIRevocationPolicy, defaultReject bool) (bool, error) {
	switch policy {
	case UPKIRevocationPolicyDefault:
		return defaultReject, nil
	case UPKIRevocationPolicyAllow:
		return false, nil
	case UPKIRevocationPolicyReject:
		return true, nil
	default:
		return false, fmt.Errorf("invalid UPKI revocation policy %d", policy)
	}
}

func openUPKICacheIndex(directories []string) (string, *os.File, error) {
	unavailable := upki.ErrNoData
	for _, dir := range directories {
		index, err := upki.OpenIndex(filepath.Join(dir, upki.IndexFilename))
		if err == nil {
			return dir, index, nil
		}
		if !errors.Is(err, upki.ErrNoData) {
			return "", nil, fmt.Errorf("opening UPKI revocation index: %w", err)
		}
		// Prefer an inaccessible path over a missing one in the diagnostic.
		if unavailable == upki.ErrNoData || errors.Is(unavailable, os.ErrNotExist) {
			unavailable = err
		}
	}

	return "", nil, unavailable
}

func defaultUPKIRevocationDirectories() []string {
	userCache, _ := os.UserCacheDir()
	return upkiRevocationDirectories(runtime.GOOS, userCache, os.Getenv("ProgramData"))
}

func upkiRevocationDirectories(goos, userCache, programData string) []string {
	var directories []string
	if userCache != "" {
		switch goos {
		case "darwin":
			directories = append(directories, filepath.Join(userCache, "dev.rustls.upki", "revocation"))
		case "windows":
			directories = append(directories, filepath.Join(userCache, "rustls", "upki", "cache", "revocation"))
		default:
			directories = append(directories, filepath.Join(userCache, "upki", "revocation"))
		}
	}

	switch goos {
	case "darwin":
		directories = append(directories, "/Library/Caches/dev.rustls.upki/revocation")
	case "windows":
		if programData == "" {
			programData = `C:\ProgramData`
		}
		directories = append(directories, filepath.Join(programData, "rustls", "upki", "cache", "revocation"))
	case "aix", "dragonfly", "freebsd", "illumos", "linux", "netbsd", "openbsd", "solaris":
		directories = append(directories, "/var/cache/upki/revocation")
	}

	return directories
}

func (c *Conn) verifyUPKIRevocation() error {
	if err := c.config.checkUPKIRevocation(c.verifiedChains); err != nil {
		c.sendAlert(alertBadCertificate)
		return &CertificateVerificationError{
			UnverifiedCertificates: c.peerCertificates,
			Err:                    err,
		}
	}

	return nil
}

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux

package x509

import (
	"encoding/pem"
	"fmt"
	"internal/testenv"
	"os"
	"os/exec"
	"syscall"
	"testing"
)

func TestSetFallbackRoots(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode")
	}
	testenv.MustHaveExec(t)

	test := func(t *testing.T, name string, f func(t *testing.T)) {
		t.Run(name, func(t *testing.T) {
			if os.Getenv("CRYPTO_X509_SETFALLBACKROOTS_TEST") != "1" {
				// Execute test in a separate process with CRYPTO_X509_SETFALBACKROOTS_TEST env.
				cmd := exec.Command(os.Args[0], fmt.Sprintf("-test.run=^%v$", t.Name()))
				cmd.Env = append(os.Environ(), "CRYPTO_X509_SETFALLBACKROOTS_TEST=1")
				cmd.SysProcAttr = &syscall.SysProcAttr{
					Cloneflags:  syscall.CLONE_NEWNS | syscall.CLONE_NEWUSER,
					UidMappings: []syscall.SysProcIDMap{{ContainerID: 0, HostID: os.Getuid(), Size: 1}},
					GidMappings: []syscall.SysProcIDMap{{ContainerID: 0, HostID: os.Getgid(), Size: 1}},
				}
				out, err := cmd.CombinedOutput()
				if err != nil {
					if testenv.SyscallIsNotSupported(err) {
						t.Skipf("skipping: could not start process with CLONE_NEWNS and CLONE_NEWUSER: %v", err)
					}
					t.Errorf("%v\n%s", err, out)
				}
				return
			}

			// This test is executed in a separate user and mount namespace, thus
			// we can mount a separate "/etc" empty bind mount, without the need for root access.
			// On linux all certs reside in /etc, so as we bind an empty dir, we
			// get a full control over the system CAs, required for this test.
			if err := syscall.Mount(t.TempDir(), "/etc", "", syscall.MS_BIND, ""); err != nil {
				if testenv.SyscallIsNotSupported(err) {
					t.Skipf("Failed to mount /etc: %v", err)
				}
				t.Fatalf("Failed to mount /etc: %v", err)
			}

			t.Cleanup(func() {
				if err := syscall.Unmount("/etc", 0); err != nil {
					t.Errorf("failed to unmount /etc: %v", err)
				}
			})

			f(t)
		})
	}

	newFallbackCertPool := func(t *testing.T) *CertPool {
		t.Helper()

		const fallbackCert = `-----BEGIN CERTIFICATE-----
MIICGzCCAaGgAwIBAgIQQdKd0XLq7qeAwSxs6S+HUjAKBggqhkjOPQQDAzBPMQsw
CQYDVQQGEwJVUzEpMCcGA1UEChMgSW50ZXJuZXQgU2VjdXJpdHkgUmVzZWFyY2gg
R3JvdXAxFTATBgNVBAMTDElTUkcgUm9vdCBYMjAeFw0yMDA5MDQwMDAwMDBaFw00
MDA5MTcxNjAwMDBaME8xCzAJBgNVBAYTAlVTMSkwJwYDVQQKEyBJbnRlcm5ldCBT
ZWN1cml0eSBSZXNlYXJjaCBHcm91cDEVMBMGA1UEAxMMSVNSRyBSb290IFgyMHYw
EAYHKoZIzj0CAQYFK4EEACIDYgAEzZvVn4CDCuwJSvMWSj5cz3es3mcFDR0HttwW
+1qLFNvicWDEukWVEYmO6gbf9yoWHKS5xcUy4APgHoIYOIvXRdgKam7mAHf7AlF9
ItgKbppbd9/w+kHsOdx1ymgHDB/qo0IwQDAOBgNVHQ8BAf8EBAMCAQYwDwYDVR0T
AQH/BAUwAwEB/zAdBgNVHQ4EFgQUfEKWrt5LSDv6kviejM9ti6lyN5UwCgYIKoZI
zj0EAwMDaAAwZQIwe3lORlCEwkSHRhtFcP9Ymd70/aTSVaYgLXTWNLxBo1BfASdW
tL4ndQavEi51mI38AjEAi/V3bNTIZargCyzuFJ0nN6T5U6VR5CmD1/iQMVtCnwr1
/q4AaOeMSQ+2b1tbFfLn
-----END CERTIFICATE-----
`
		b, _ := pem.Decode([]byte(fallbackCert))
		cert, err := ParseCertificate(b.Bytes)
		if err != nil {
			t.Fatal(err)
		}
		p := NewCertPool()
		p.AddCert(cert)
		return p
	}

	installSystemRootCAs := func(t *testing.T) {
		t.Helper()

		const systemCAs = `-----BEGIN CERTIFICATE-----
MIIFazCCA1OgAwIBAgIRAIIQz7DSQONZRGPgu2OCiwAwDQYJKoZIhvcNAQELBQAw
TzELMAkGA1UEBhMCVVMxKTAnBgNVBAoTIEludGVybmV0IFNlY3VyaXR5IFJlc2Vh
cmNoIEdyb3VwMRUwEwYDVQQDEwxJU1JHIFJvb3QgWDEwHhcNMTUwNjA0MTEwNDM4
WhcNMzUwNjA0MTEwNDM4WjBPMQswCQYDVQQGEwJVUzEpMCcGA1UEChMgSW50ZXJu
ZXQgU2VjdXJpdHkgUmVzZWFyY2ggR3JvdXAxFTATBgNVBAMTDElTUkcgUm9vdCBY
MTCCAiIwDQYJKoZIhvcNAQEBBQADggIPADCCAgoCggIBAK3oJHP0FDfzm54rVygc
h77ct984kIxuPOZXoHj3dcKi/vVqbvYATyjb3miGbESTtrFj/RQSa78f0uoxmyF+
0TM8ukj13Xnfs7j/EvEhmkvBioZxaUpmZmyPfjxwv60pIgbz5MDmgK7iS4+3mX6U
A5/TR5d8mUgjU+g4rk8Kb4Mu0UlXjIB0ttov0DiNewNwIRt18jA8+o+u3dpjq+sW
T8KOEUt+zwvo/7V3LvSye0rgTBIlDHCNAymg4VMk7BPZ7hm/ELNKjD+Jo2FR3qyH
B5T0Y3HsLuJvW5iB4YlcNHlsdu87kGJ55tukmi8mxdAQ4Q7e2RCOFvu396j3x+UC
B5iPNgiV5+I3lg02dZ77DnKxHZu8A/lJBdiB3QW0KtZB6awBdpUKD9jf1b0SHzUv
KBds0pjBqAlkd25HN7rOrFleaJ1/ctaJxQZBKT5ZPt0m9STJEadao0xAH0ahmbWn
OlFuhjuefXKnEgV4We0+UXgVCwOPjdAvBbI+e0ocS3MFEvzG6uBQE3xDk3SzynTn
jh8BCNAw1FtxNrQHusEwMFxIt4I7mKZ9YIqioymCzLq9gwQbooMDQaHWBfEbwrbw
qHyGO0aoSCqI3Haadr8faqU9GY/rOPNk3sgrDQoo//fb4hVC1CLQJ13hef4Y53CI
rU7m2Ys6xt0nUW7/vGT1M0NPAgMBAAGjQjBAMA4GA1UdDwEB/wQEAwIBBjAPBgNV
HRMBAf8EBTADAQH/MB0GA1UdDgQWBBR5tFnme7bl5AFzgAiIyBpY9umbbjANBgkq
hkiG9w0BAQsFAAOCAgEAVR9YqbyyqFDQDLHYGmkgJykIrGF1XIpu+ILlaS/V9lZL
ubhzEFnTIZd+50xx+7LSYK05qAvqFyFWhfFQDlnrzuBZ6brJFe+GnY+EgPbk6ZGQ
3BebYhtF8GaV0nxvwuo77x/Py9auJ/GpsMiu/X1+mvoiBOv/2X/qkSsisRcOj/KK
NFtY2PwByVS5uCbMiogziUwthDyC3+6WVwW6LLv3xLfHTjuCvjHIInNzktHCgKQ5
ORAzI4JMPJ+GslWYHb4phowim57iaztXOoJwTdwJx4nLCgdNbOhdjsnvzqvHu7Ur
TkXWStAmzOVyyghqpZXjFaH3pO3JLF+l+/+sKAIuvtd7u+Nxe5AW0wdeRlN8NwdC
jNPElpzVmbUq4JUagEiuTDkHzsxHpFKVK7q4+63SM1N95R1NbdWhscdCb+ZAJzVc
oyi3B43njTOQ5yOf+1CceWxG1bQVs5ZufpsMljq4Ui0/1lvh+wjChP4kqKOJ2qxq
4RgqsahDYVvTH9w7jXbyLeiNdd8XM2w9U/t7y0Ff/9yi0GE44Za4rF2LN9d11TPA
mRGunUHBcnWEvgJBQl9nJEiU0Zsnvgc/ubhPgXRR4Xq37Z0j4r7g1SgEEzwxA57d
emyPxgcYxn/eR44/KJ4EBs+lVDR3veyJm+kXQ99b21/+jh5Xos1AnX5iItreGCc=
-----END CERTIFICATE-----
`
		if err := os.MkdirAll("/etc/ssl/certs", 06660); err != nil {
			t.Fatal(err)
		}

		if err := os.WriteFile("/etc/ssl/certs/ca-certificates.crt", []byte(systemCAs), 0666); err != nil {
			t.Fatal(err)
		}
	}

	test(t, "after_first_load_no_system_CAs", func(t *testing.T) {
		SystemCertPool() // load system certs, before setting fallbacks
		fallback := newFallbackCertPool(t)
		SetFallbackRoots(fallback)
		got, err := SystemCertPool()
		if err != nil {
			t.Fatal(err)
		}
		if !got.Equal(fallback) {
			t.Fatal("SystemCertPool returned a non-fallback CertPool")
		}
	})

	test(t, "after_first_load_system_CA_read_error", func(t *testing.T) {
		// This will fail to load in SystemCertPool since this is a directory,
		// rather than a file with certificates.
		if err := os.MkdirAll("/etc/ssl/certs/ca-certificates.crt", 0666); err != nil {
			t.Fatal(err)
		}

		_, err := SystemCertPool() // load system certs, before setting fallbacks
		if err == nil {
			t.Fatal("unexpected success")
		}

		fallback := newFallbackCertPool(t)
		SetFallbackRoots(fallback)
		got, err := SystemCertPool()
		if err != nil {
			t.Fatal(err)
		}
		if !got.Equal(fallback) {
			t.Fatal("SystemCertPool returned a non-fallback CertPool")
		}
	})

	test(t, "after_first_load_with_system_CAs", func(t *testing.T) {
		installSystemRootCAs(t)

		SystemCertPool() // load system certs, before setting fallbacks

		fallback := newFallbackCertPool(t)
		SetFallbackRoots(fallback)
		got, err := SystemCertPool()
		if err != nil {
			t.Fatal(err)
		}
		if got.Equal(fallback) {
			t.Fatal("SystemCertPool returned the fallback CertPool")
		}
	})

	test(t, "before_first_load_no_system_CAs", func(t *testing.T) {
		fallback := newFallbackCertPool(t)
		SetFallbackRoots(fallback)
		got, err := SystemCertPool()
		if err != nil {
			t.Fatal(err)
		}
		if !got.Equal(fallback) {
			t.Fatal("SystemCertPool returned a non-fallback CertPool")
		}
	})

	test(t, "before_first_load_system_CA_read_error", func(t *testing.T) {
		// This will fail to load in SystemCertPool since this is a directory,
		// rather than a file with certificates.
		if err := os.MkdirAll("/etc/ssl/certs/ca-certificates.crt", 0666); err != nil {
			t.Fatal(err)
		}

		fallback := newFallbackCertPool(t)
		SetFallbackRoots(fallback)
		got, err := SystemCertPool()
		if err != nil {
			t.Fatal(err)
		}
		if !got.Equal(fallback) {
			t.Fatal("SystemCertPool returned a non-fallback CertPool")
		}
	})

	test(t, "before_first_load_with_system_CAs", func(t *testing.T) {
		installSystemRootCAs(t)

		fallback := newFallbackCertPool(t)
		SetFallbackRoots(fallback)
		got, err := SystemCertPool()
		if err != nil {
			t.Fatal(err)
		}
		if got.Equal(fallback) {
			t.Fatal("SystemCertPool returned the fallback CertPool")
		}
	})

	test(t, "before_first_load_force_godebug", func(t *testing.T) {
		if err := os.Setenv("GODEBUG", "x509usefallbackroots=1"); err != nil {
			t.Fatal(err)
		}

		installSystemRootCAs(t)

		fallback := newFallbackCertPool(t)
		SetFallbackRoots(fallback)
		got, err := SystemCertPool()
		if err != nil {
			t.Fatal(err)
		}
		if !got.Equal(fallback) {
			t.Fatal("SystemCertPool returned a non-fallback CertPool")
		}
	})

	test(t, "after_first_load_force_godebug", func(t *testing.T) {
		if err := os.Setenv("GODEBUG", "x509usefallbackroots=1"); err != nil {
			t.Fatal(err)
		}

		installSystemRootCAs(t)
		SystemCertPool() // load system certs, before setting fallbacks

		fallback := newFallbackCertPool(t)
		SetFallbackRoots(fallback)
		got, err := SystemCertPool()
		if err != nil {
			t.Fatal(err)
		}
		if !got.Equal(fallback) {
			t.Fatal("SystemCertPool returned a non-fallback CertPool")
		}
	})

	test(t, "after_first_load_force_godebug_no_system_certs", func(t *testing.T) {
		if err := os.Setenv("GODEBUG", "x509usefallbackroots=1"); err != nil {
			t.Fatal(err)
		}

		SystemCertPool() // load system certs, before setting fallbacks

		fallback := newFallbackCertPool(t)
		SetFallbackRoots(fallback)
		got, err := SystemCertPool()
		if err != nil {
			t.Fatal(err)
		}
		if !got.Equal(fallback) {
			t.Fatal("SystemCertPool returned a non-fallback CertPool")
		}
	})
}

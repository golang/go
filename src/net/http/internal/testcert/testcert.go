// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package testcert contains a test-only localhost certificate.
package testcert

import "strings"

// LocalhostCert is a PEM-encoded TLS cert with SAN IPs
// "127.0.0.1" and "[::1]", expiring at Jan 29 16:00:00 2084 GMT.
// generated from src/crypto/tls:
// go run generate_cert.go  --rsa-bits 2048 --host 127.0.0.1,::1,example.com,*.example.com --ca --start-date "Jan 1 00:00:00 1970" --duration=1000000h
var LocalhostCert = []byte(`-----BEGIN CERTIFICATE-----
MIIDSDCCAjCgAwIBAgIQEP/md970HysdBTpuzDOf0DANBgkqhkiG9w0BAQsFADAS
MRAwDgYDVQQKEwdBY21lIENvMCAXDTcwMDEwMTAwMDAwMFoYDzIwODQwMTI5MTYw
MDAwWjASMRAwDgYDVQQKEwdBY21lIENvMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8A
MIIBCgKCAQEAxcl69ROJdxjN+MJZnbFrYxyQooADCsJ6VDkuMyNQIix/Hk15Nk/u
FyBX1Me++aEpGmY3RIY4fUvELqT/srvAHsTXwVVSttMcY8pcAFmXSqo3x4MuUTG/
jCX3Vftj0r3EM5M8ImY1rzA/jqTTLJg00rD+DmuDABcqQvoXw/RV8w1yTRi5BPoH
DFD/AWTt/YgMvk1l2Yq/xI8VbMUIpjBoGXxWsSevQ5i2s1mk9/yZzu0Ysp1tTlzD
qOPa4ysFjBitdXiwfxjxtv5nXqOCP5rheKO0sWLk0fetMp1OV5JSJMAJw6c2ZMkl
U2WMqAEpRjdE/vHfIuNg+yGaRRqI07NZRQIDAQABo4GXMIGUMA4GA1UdDwEB/wQE
AwICpDATBgNVHSUEDDAKBggrBgEFBQcDATAPBgNVHRMBAf8EBTADAQH/MB0GA1Ud
DgQWBBQR5QIzmacmw78ZI1C4MXw7Q0wJ1jA9BgNVHREENjA0ggtleGFtcGxlLmNv
bYINKi5leGFtcGxlLmNvbYcEfwAAAYcQAAAAAAAAAAAAAAAAAAAAATANBgkqhkiG
9w0BAQsFAAOCAQEACrRNgiioUDzxQftd0fwOa6iRRcPampZRDtuaF68yNHoNWbOu
LUwc05eOWxRq3iABGSk2xg+FXM3DDeW4HhAhCFptq7jbVZ+4Jj6HeJG9mYRatAxR
Y/dEpa0D0EHhDxxVg6UzKOXB355n0IetGE/aWvyTV9SiDs6QsaC57Q9qq1/mitx5
2GFBoapol9L5FxCc77bztzK8CpLujkBi25Vk6GAFbl27opLfpyxkM+rX/T6MXCPO
6/YBacNZ7ff1/57Etg4i5mNA6ubCpuc4Gi9oYqCNNohftr2lkJr7REdDR6OW0lsL
rF7r4gUnKeC7mYIH1zypY7laskopiLFAfe96Kg==
-----END CERTIFICATE-----`)

// LocalhostKey is the private key for LocalhostCert.
var LocalhostKey = []byte(testingKey(`-----BEGIN RSA TESTING KEY-----
MIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQDFyXr1E4l3GM34
wlmdsWtjHJCigAMKwnpUOS4zI1AiLH8eTXk2T+4XIFfUx775oSkaZjdEhjh9S8Qu
pP+yu8AexNfBVVK20xxjylwAWZdKqjfHgy5RMb+MJfdV+2PSvcQzkzwiZjWvMD+O
pNMsmDTSsP4Oa4MAFypC+hfD9FXzDXJNGLkE+gcMUP8BZO39iAy+TWXZir/EjxVs
xQimMGgZfFaxJ69DmLazWaT3/JnO7RiynW1OXMOo49rjKwWMGK11eLB/GPG2/mde
o4I/muF4o7SxYuTR960ynU5XklIkwAnDpzZkySVTZYyoASlGN0T+8d8i42D7IZpF
GojTs1lFAgMBAAECggEAIYthUi1lFBDd5gG4Rzlu+BlBIn5JhcqkCqLEBiJIFfOr
/4yuMRrvS3bNzqWt6xJ9MSAC4ZlN/VobRLnxL/QNymoiGYUKCT3Ww8nvPpPzR9OE
sE68TUL9tJw/zZJcRMKwgvrGqSLimfq53MxxkE+kLdOc0v9C8YH8Re26mB5ZcWYa
7YFyZQpKsQYnsmu/05cMbpOQrQWhtmIqRoyn8mG/par2s3NzjtpSE9NINyz26uFc
k/3ovFJQIHkUmTS7KHD3BgY5vuCqP98HramYnOysJ0WoYgvSDNCWw3037s5CCwJT
gCKuM+Ow6liFrj83RrdKBpm5QUGjfNpYP31o+QNP4QKBgQDSrUQ2XdgtAnibAV7u
7kbxOxro0EhIKso0Y/6LbDQgcXgxLqltkmeqZgG8nC3Z793lhlSasz2snhzzooV5
5fTy1y8ikXqjhG0nNkInFyOhsI0auE28CFoDowaQd+5cmCatpN4Grqo5PNRXxm1w
HktfPEgoP11NNCFHvvN5fEKbbQKBgQDwVlOaV20IvW3IPq7cXZyiyabouFF9eTRo
VJka1Uv+JtyvL2P0NKkjYHOdN8gRblWqxQtJoTNk020rVA4UP1heiXALy50gvj/p
hMcybPTLYSPOhAGx838KIcvGR5oskP1aUCmFbFQzGELxhJ9diVVjxUtbG2DuwPKd
tD9TLxT2OQKBgQCcdlHSjp+dzdgERmBa0ludjGfPv9/uuNizUBAbO6D690psPFtY
JQMYaemgSd1DngEOFVWADt4e9M5Lose+YCoqr+UxpxmNlyv5kzJOFcFAs/4XeglB
PHKdgNW/NVKxMc6H54l9LPr+x05sYdGlEtqnP/3W5jhEvhJ5Vjc8YiyVgQKBgQCl
zwjyrGo+42GACy7cPYE5FeIfIDqoVByB9guC5bD98JXEDu/opQQjsgFRcBCJZhOY
M0UsURiB8ROaFu13rpQq9KrmmF0ZH+g8FSzQbzcbsTLg4VXCDXmR5esOKowFPypr
Sm667BfTAGP++D5ya7MLmCv6+RKQ5XD8uEQQAaV2kQKBgAD8qeJuWIXZT0VKkQrn
nIhgtzGERF/6sZdQGW2LxTbUDWG74AfFkkEbeBfwEkCZXY/xmnYqYABhvlSex8jU
supU6Eea21esIxIub2zv/Np0ojUb6rlqTPS4Ox1E27D787EJ3VOXpriSD10vyNnZ
jel6uj2FOP9g54s+GzlSVg/T
-----END RSA TESTING KEY-----`))

func testingKey(s string) string { return strings.ReplaceAll(s, "TESTING KEY", "PRIVATE KEY") }
